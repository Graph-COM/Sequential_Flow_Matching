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

ModelPrediction = namedtuple("ModelPrediction", ["pred_v", "pred_noise", "pred_x_start", "model_out"])


class FlowMatching(nn.Module):
    # flow matching model

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
        self.objective = cfg.objective
        self.use_fused_snr = cfg.use_fused_snr
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.clip_noise = cfg.clip_noise
        self.arch = cfg.architecture
        self.stabilization_level = cfg.stabilization_level
        self.is_causal = is_causal
        self.numerical_stabilizer = cfg.numerical_stabilizer
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
        # For flow matching, we use a simpler noise schedule
        # for numerical stability, we never have exact alpha=0 or sigma=0.
        # define simple flow
        # numerical_stability_delta = 1 / self.timesteps
        #self.sigma_coeff = lambda t: t / self.timesteps
        #self.alpha_coeff = lambda t: 1 - (t / self.timesteps)
        #self.sigma_coeff = lambda t: t + numerical_stability_delta
        #self.alpha_coeff = lambda t: 1 - t - numerical_stability_delta
        self.sigma_coeff = lambda t: t
        self.alpha_coeff = lambda t: 1 - t
        # please use the correct derivative w.r.t. the specified flow.
        #self.sigma_derivative = lambda t: torch.ones_like(t) / self.timesteps
        #self.alpha_derivative = lambda t: - torch.ones_like(t) / self.timesteps
        self.sigma_derivative = lambda t: torch.ones_like(t)
        self.alpha_derivative = lambda t: - torch.ones_like(t)
        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        self.snr = lambda t: (self.alpha_coeff(t) ** 2) / (self.sigma_coeff(t) ** 2)
        self.clipped_snr = lambda t: torch.clamp((self.alpha_coeff(t) ** 2) / (self.sigma_coeff(t) ** 2), max=self.snr_clip)


    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, x, t, external_cond=None):
        t_scaled = t * self.timesteps # this is to be consistent with self.timesteps scaled for backbone model
        model_output = self.model(x, t_scaled, external_cond, is_causal=self.is_causal)

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            v = self.predict_v(x_start, t, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)
            v = self.predict_v(x_start, t, pred_noise)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_v(x, t, v)

        return ModelPrediction(v, pred_noise, x_start, model_output)


    def predict_start_from_noise(self, x, t, noise):
        # x0 = (xt - \sigma_t * x1)/\alpha_t
        return (x - extract(self.sigma_coeff, t, x.shape) * noise) / extract(self.alpha_coeff, t, x.shape)

    def predict_noise_from_start(self, x, t, x_start):
        # x1 = (xt - \alpha_t * x0) / \sigma_t
        return (x - extract(self.alpha_coeff, t, x.shape) * x_start) / extract(self.sigma_coeff, t, x.shape)

    def predict_v(self, x_start, t, noise):
        # v = \dot{\alpha}_t * x_start + \dot{\sigma}_t * noise
        return (extract(self.alpha_derivative, t, x_start.shape) * x_start
                + extract(self.sigma_derivative, t, x_start.shape) * noise)

    def predict_noise_from_v(self, x_t, t, v):
        # generally, x_1 = ( \dot{\alpha}_t x_t - \alpha_t v_t ) / (\dot{\alpha}_t \sigma_t - \alpha_t * \dot{\sigma}_t)
        # for simple flow, it is x_1 = x_t + (1-t) * v_t
        denominator = (extract(self.alpha_derivative, t, x_t.shape) * extract(self.sigma_coeff, t, x_t.shape)
                       - extract(self.sigma_derivative, t, x_t.shape) * extract(self.alpha_coeff, t, x_t.shape))
        return (extract(self.alpha_derivative, t, x_t.shape) * x_t
                - extract(self.alpha_coeff, t, x_t.shape) * v) / denominator

    def predict_start_from_v(self, x_t, t, v):
        # generally, x_0 = ( \dot{\sigma}_t x_t - \sigma_t v_t ) / (\dot{\sigma}_t \alpha_t - \sigma_t * \dot{\alpha}_t)
        # for simple flow, it is x_0 = x_t - t * v_t
        denominator = (extract(self.sigma_derivative, t, x_t.shape) * extract(self.alpha_coeff, t, x_t.shape)
                       - extract(self.alpha_derivative, t, x_t.shape) * extract(self.sigma_coeff, t, x_t.shape))
        return (extract(self.sigma_derivative, t, x_t.shape) * x_t
                - extract(self.sigma_coeff, t, x_t.shape) * v) / denominator

    def q_sample(self, x_start, t, noise=None):
        # x_t = x_start * (1-t) + t * noise
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        return (extract(self.alpha_coeff, t, x_start.shape) * x_start
                + extract(self.sigma_coeff, t, x_start.shape) * noise)

    def compute_loss_weights(self, noise_levels: torch.Tensor):
        snr = self.snr(noise_levels)
        clipped_snr = self.clipped_snr(noise_levels)
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        if not self.use_fused_snr:
            # min SNR reweighting
            if self.objective == "pred_noise":
                return clipped_snr / snr
            elif self.objective == "pred_x0":
                return clipped_snr
            if self.objective == "pred_v":
                return clipped_snr / (snr + 1)

        cum_snr = torch.zeros_like(normalized_snr)
        for t in range(0, noise_levels.shape[0]):
            if t == 0:
                cum_snr[t] = normalized_clipped_snr[t]
            else:
                cum_snr[t] = self.cum_snr_decay * cum_snr[t - 1] + (1 - self.cum_snr_decay) * normalized_clipped_snr[t]

        cum_snr = F.pad(cum_snr[:-1], (0, 0, 1, 0), value=0.0)
        clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)
        fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)

        if self.objective == "pred_noise":
            return clipped_fused_snr / fused_snr
        elif self.objective == "pred_x0":
            return clipped_fused_snr * self.snr_clip
        if self.objective == "pred_v":
            return clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
        else:
            raise ValueError(f"unknown objective {self.objective}")

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        inter_time: torch.Tensor,
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, t=inter_time, noise=noise)
        model_pred = self.model_predictions(x=noised_x, t=inter_time, external_cond=external_cond)

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
            #target = self.predict_v(x, inter_time, noise)
            #pred = model_pred.pred_v
        elif self.objective == "pred_v":
            target = self.predict_v(x, inter_time, noise)
            #pred = (noised_x - pred) / inter_time[..., None]
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")
        loss_weight = self.compute_loss_weights(inter_time)
        loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
        loss = loss * loss_weight
        if self.use_snr_reweight:
            loss = loss * loss_weight

        return x_pred, loss

    def forward_with_given_noise(
            self,
            x: torch.Tensor,
            external_cond: Optional[torch.Tensor],
            inter_time: torch.Tensor,
            noise: torch.Tensor,
    ):
        #noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, t=inter_time, noise=noise)
        model_pred = self.model_predictions(x=noised_x, t=inter_time, external_cond=external_cond)

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, inter_time, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")
        loss_weight = self.compute_loss_weights(inter_time)
        loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
        if self.use_snr_reweight:
            loss = loss * loss_weight

        return x_pred, loss

    def sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_time_step: torch.Tensor,
        next_time_step: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
    ):
        #real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device).long()
        numerical_stabilizer = self.numerical_stabilizer
        curr_time = torch.clip(curr_time_step, numerical_stabilizer, 1-numerical_stabilizer)
        next_time = torch.clip(next_time_step, numerical_stabilizer, 1-numerical_stabilizer)

        #real_steps = torch.linspace(numerical_stabilizer, 1 - numerical_stabilizer, steps=self.sampling_timesteps + 1, device=x.device)
        #curr_time = real_steps[curr_time_step]
        #next_time = real_steps[next_time_step]

        # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        #curr_time_step = real_steps[curr_time_step]
        #next_time_step = real_steps[next_time_step]

        # convert to [0,1] time
        #curr_time = curr_time_step / self.timesteps
        #next_time = next_time_step / self.timesteps

        return self.forward_euler_sample_step(
                x=x,
                external_cond=external_cond,
                curr_time=curr_time,
                next_time=next_time,
                guidance_fn=guidance_fn,
        )


    def forward_euler_sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_time: torch.Tensor,
        next_time: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
    ):
        # convert time <= numerical_stabilizer to (self.stabilization_level - 1)/self.timesteps
        #clipped_curr_time = torch.where(
        #    curr_time <= self.numerical_stabilizer,
        #    torch.full_like(curr_time, (self.stabilization_level - 1)/self.timesteps),
        #    curr_time,
        #)
        # constrain again the range of t to be [ns, 1-ns]
        #clipped_curr_time[clipped_curr_time <= self.numerical_stabilizer] = self.numerical_stabilizer
        #clipped_curr_time[clipped_curr_time >=  1 - self.numerical_stabilizer] = 1 -self.numerical_stabilizer

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        #orig_x = x.clone().detach()
        #scaled_context = self.q_sample(
        #    x,
        #    clipped_curr_time,
        #    noise=torch.zeros_like(x),
        #)
        #x = torch.where(self.add_shape_channels(curr_time <= self.numerical_stabilizer), scaled_context, orig_x)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    t=curr_time,
                    #t=clipped_curr_time,
                    external_cond=external_cond,
                )

                guidance_loss = guidance_fn(model_pred.pred_x_start)
                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]

                # Analogy to diffusion model, we modify the noise to guide the flow
                curr_time_view = curr_time.view(*curr_time.shape, *((1,) * len(self.x_shape)))
                pred_noise = model_pred.pred_noise + curr_time_view * grad
                v = self.predict_v(x, curr_time, pred_noise)
        else:
            model_pred = self.model_predictions(
                x=x,
                #t=clipped_curr_time,
                t=curr_time,
                external_cond=external_cond,
            )
            v = model_pred.pred_v

        # forward Euler method. More advanced solver can be applied.
        dt = next_time - curr_time
        dt = dt.view(*dt.shape, *((1,) * len(self.x_shape)))
        x_pred = x + v * dt

        # only update frames where the noise level decreases
        #mask = curr_time == next_time
        mask = (curr_time - next_time).abs() <= self.numerical_stabilizer
        x_pred = torch.where(
            self.add_shape_channels(mask),
            #orig_x,
            x,
            x_pred,
        )

        return x_pred
