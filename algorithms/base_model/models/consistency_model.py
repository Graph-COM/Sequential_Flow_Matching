from typing import Optional, Callable
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .unet3d import Unet3D
from .transformer import Transformer
from .utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract, EinopsWrapper

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


class ConsistencyModel(nn.Module):
    # https://arxiv.org/abs/2406.14548

    def __init__(
        self,
        x_shape: torch.Size,
        external_cond_dim: int,
        is_causal: bool,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.x_shape = x_shape
        self.external_cond_dim = external_cond_dim
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.objective = cfg.objective
        self.use_fused_snr = cfg.use_fused_snr
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise
        self.arch = cfg.architecture
        self.stabilization_level = cfg.stabilization_level
        self.is_causal = is_causal
        self.use_snr_reweight = cfg.use_snr_reweight
        self.snr_reweight = cfg.snr_reweight

        self._build_model()
        self._build_buffer()

    def _build_model(self):
        x_channel = self.x_shape[0]
        if len(self.x_shape) == 3:
            # video
            attn_resolutions = [self.arch.resolution // res for res in list(self.arch.attn_resolutions)]
            self.model = EinopsWrapper(
                from_shape="f b c h w",
                to_shape="b c f h w",
                module=Unet3D(
                    dim=self.arch.network_size,
                    attn_dim_head=self.arch.attn_dim_head,
                    attn_heads=self.arch.attn_heads,
                    dim_mults=self.arch.dim_mults,
                    attn_resolutions=attn_resolutions,
                    use_linear_attn=self.arch.use_linear_attn,
                    channels=x_channel,
                    out_dim=x_channel,
                    external_cond_dim=self.external_cond_dim,
                    is_causal=self.is_causal,
                    use_init_temporal_attn=self.arch.use_init_temporal_attn,
                    time_emb_type=self.arch.time_emb_type,
                ),
            )
        elif len(self.x_shape) == 1:
            self.model = Transformer(
                x_dim=x_channel,
                external_cond_dim=self.external_cond_dim,
                size=self.arch.network_size,
                num_layers=self.arch.num_layers,
                nhead=self.arch.attn_heads,
                dim_feedforward=self.arch.dim_feedforward,
            )
        else:
            raise ValueError(f"unsupported input shape {self.x_shape}")

    def _build_buffer(self):
        if self.beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {self.beta_schedule}")

        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        # register_buffer(
        #     "p2_loss_weight",
        #     (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
        #     ** -self.p2_loss_weight_gamma,
        # )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        clipped_snr = snr.clone()
        if self.snr_reweight == 'max_snr':
            # max_snr means max(snr, snr_clip) = clip(snr, min=snr_clip)
            clipped_snr.clamp_(min=self.snr_clip)
        elif self.snr_reweight == 'min_snr':
            # min_snr means min(snr, snr_clip) = clip(snr, max=snr_clip)
            clipped_snr.clamp_(max=self.snr_clip)
        else:
            pass

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, x, t, external_cond=None, pad_mask=None):
        model_output = self.model(x, t, external_cond, is_causal=self.is_causal, mask=pad_mask)
        # boundary condition
        t_norm_expand = t.view(list(t.shape) + [1] * len(self.x_shape))
        t_norm_expand = t_norm_expand / self.timesteps
        model_output = x / (1 + t_norm_expand.square()) + t_norm_expand * model_output / (torch.sqrt(1 + t_norm_expand.square()))

        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start, model_output)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        noise_levels,
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noise_levels_t, noise_levels_r = noise_levels

        noised_xt = self.q_sample(x_start=x, t=noise_levels_t, noise=noise)
        noised_xr = self.q_sample(x_start=x, t=noise_levels_r, noise=noise)
        model_pred_t = self.model_predictions(x=noised_xt, t=noise_levels_t, external_cond=external_cond)
        model_pred_r = self.model_predictions(x=noised_xr, t=noise_levels_r, external_cond=external_cond)

        x_pred_t = model_pred_t.pred_x_start
        x_pred_r = model_pred_r.pred_x_start

        # consistency loss: pseudo-Huber metric
        eps = 1e-1
        loss = F.mse_loss(x_pred_t, x_pred_r.detach(), reduction="none")
        loss = torch.sqrt(loss + eps**2) - eps
        #loss = F.mse_loss(x_pred_t, x_pred_r.detach(), reduction='none')

        return x_pred_t, loss

    def sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
    ):
        # in the current code curr_noise_level already represents real steps
        #real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device).long()

        # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        #curr_noise_level = real_steps[curr_noise_level]
        #next_noise_level = real_steps[next_noise_level]

        # in case we want to add stabilization noises
        curr_noise_level[curr_noise_level == 0] = -1
        next_noise_level[next_noise_level == 0] = -1

        return self.sample_step(
            x=x,
            external_cond=external_cond,
            curr_noise_level=curr_noise_level,
            next_noise_level=next_noise_level,
            guidance_fn=guidance_fn,
        )

    def sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
    ):
        # convert noise level -1 to self.stabilization_level - 1
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        if guidance_fn is not None:
            raise NotImplementedError('guidance function not implemented for consistency model')
        else:
            model_pred = self.model_predictions(
                x=x,
                t=clipped_curr_noise_level,
                external_cond=external_cond,
            )
            x_pred = model_pred.pred_x_start

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )

        return x_pred
