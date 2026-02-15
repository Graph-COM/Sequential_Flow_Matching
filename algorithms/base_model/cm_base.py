"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.consistency_model import ConsistencyModel


class ConsistencyModelBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.prediction_horizon = cfg.get('prediction_horizon')
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim
        self.causal = cfg.causal
        self.n_frames = cfg.get('n_frames') if cfg.get('n_frames') is not None else cfg.get('episode_len') + cfg.frame_stack
        self.n_tokens = self.n_frames // cfg.frame_stack
        self.training_max_steps = cfg.get('training_max_steps')

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise

        self.validation_step_outputs = []
        super().__init__(cfg)

    def _build_model(self):
        self.diffusion_model = ConsistencyModel(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
        )
        if self.cfg.data_mean is not None and self.cfg.data_std is not None:
            self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def forward(self, xs: torch.Tensor, conditions: torch.Tensor, masks: torch.Tensor, context_masks=None) -> torch.Tensor:
        return self.diffusion_model(xs, conditions, self._generate_noise_levels(xs, masks=masks,
                                                                                        context_masks=context_masks))

    def model_sampling(self, xs_context, conditions, prediction_window, n_frames, batch_size, diff_hist=False,
                       guidance_fn=None, pad_tokens=0):
        return self.diffusion_forcing_sampling(
            xs_context, conditions, prediction_window, n_frames, batch_size, diff_hist=diff_hist,
            guidance_fn=guidance_fn, pad_tokens=pad_tokens)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError('Implement training_step in task class')

    def diffusion_forcing_sampling(self, xs_context, conditions, prediction_window, n_frames, batch_size,
                                   diff_hist=False, guidance_fn=None, pad_tokens=0):
        xs_pred = xs_context.clone()
        xs_pred_hist = [[] for _ in range(len(prediction_window))] if diff_hist else None
        start_frame = prediction_window[0]
        curr_frame = prediction_window[0]

        pbar = tqdm(total= self.sampling_timesteps * len(prediction_window), initial=0,
                    desc="Diffusion sampling (stacked frame %d to %d)" % (prediction_window[0], prediction_window[-1]))
        while curr_frame <= prediction_window[-1]:
            if self.chunk_size > 0:
                #horizon = min(n_frames - curr_frame, self.chunk_size)
                horizon = min(prediction_window[-1] - curr_frame + 1, self.chunk_size)
            else:
                #horizon = n_frames - curr_frame
                horizon = prediction_window[-1] - curr_frame + 1
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            # note that we use real time step instead of sampling time step as in diffusion forcing
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            #if pad_tokens > 0:
                #pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
                #xs_pred = torch.cat([xs_pred, pad], 0)

            # sliding window: only input the last n_tokens frames
            #start_frame = max(0, curr_frame + horizon - len(pre))

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64),
                     scheduling_matrix[m],
                     np.array([self.diffusion_model.timesteps-1] * pad_tokens, dtype=np.int64),
                     )
                )[:, None].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                        np.array([self.diffusion_model.timesteps-1] * pad_tokens, dtype=np.int64),
                    )
                )[:, None].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                #xs_pred[start_frame:] = self.diffusion_model.sample_step(
                #xs_pred[start_frame:],
                #conditions[start_frame: curr_frame + horizon],
                #from_noise_levels[start_frame:],
                #to_noise_levels[start_frame:],
                #)
                xs_pred = self.diffusion_model.sample_step(xs_pred, conditions[:curr_frame+horizon] if conditions is not None else None,
                                                           from_noise_levels, to_noise_levels, guidance_fn=guidance_fn)

                #if diff_hist:
                    #for t in range(curr_frame, curr_frame + horizon):
                        #xs_pred_hist[t-start_frame].append(xs_pred[t])

                pbar.update(horizon)
                pbar.set_postfix(curr_frame=curr_frame, m=m)
            curr_frame += horizon
        return xs_pred[prediction_window].clone(), xs_pred_hist

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None,
                               context_masks: Optional[torch.tensor] = None):
        """
        Generate noise levels for consistency training.
        """
        num_frames, batch_size, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_all":  # entirely random noise levels
                # continuous time sampling
                #start_end_time = torch.rand((num_frames, batch_size, 2), device=xs.device)
                dim = (num_frames, batch_size, 1)
                start_end_time = self.sample_t_r(dim, xs.device)
                # discrete time sampling
                #inter_time = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device) / self.timesteps
                #inter_time = torch.rand((num_frames, batch_size), device=xs.device) * self.timesteps
            case "uniform":
                dim = (1, batch_size, 1)
                #start_end_time = torch.rand((1, batch_size, 2), device=xs.device)
                start_end_time = self.sample_t_r(dim, xs.device)
                start_end_time = start_end_time.repeat(num_frames, 1, 1)

        start_end_time = torch.clamp(start_end_time, 0, 999)
        start_time = start_end_time[..., 0]
        end_time = start_end_time[..., 1]

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            start_time = torch.where(discard, torch.full_like(start_time, self.timesteps - 1), start_time)
            end_time = torch.where(discard, torch.full_like(end_time, self.timesteps - 1), end_time)

        if context_masks is not None:
            # for frames that are context, treat as zero noise
            discard = torch.all(~rearrange(context_masks.bool(), "(t fs) b-> t b fs", fs=self.frame_stack), -1)
            start_time = torch.where(discard, torch.zeros_like(start_time), start_time)
            end_time = torch.where(discard, torch.zeros_like(end_time), end_time)

        return start_time, end_time

    def _generate_scheduling_matrix(self, horizon: int):
        match self.cfg.scheduling_matrix:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                #return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
                return np.linspace(self.diffusion_model.timesteps-1, 0, self.sampling_timesteps+1).astype(
                    np.int64)[:, None].repeat(horizon, axis=1)
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
            #case "trapezoid":
                #return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        #for m in range(height):
        for t in range(horizon):
            # scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
            scheduling_matrix[:int(t * uncertainty_scale), t] = self.diffusion_model.timesteps-1
            scheduling_matrix[int(t*uncertainty_scale):self.sampling_timesteps + 1 + int(t * uncertainty_scale), t] = \
                np.linspace(self.diffusion_model.timesteps-1, 0,
                            self.sampling_timesteps + 1).astype(np.int64)
        return scheduling_matrix
        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    #def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        #height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        #scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        #for m in range(height):
            #for t in range((horizon + 1) // 2):
                #scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                #scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def sample_t_r(self, dim, device):
        if self.cfg.noise_level_dist == 'uniform':
            t = torch.randint(0, self.timesteps, dim, device=device)
        elif self.cfg.noise_level_dist == 'log_norm':
            mu, sigma = self.cfg.noise_level_mean, self.cfg.noise_level_std
            normal_samples = torch.randn(dim, device=device) * sigma + mu
            t = 1 / (1 + torch.exp(-normal_samples))  # Apply sigmoid
            t =  (t * self.timesteps).int()
        # sample r based on t and iter
        steps = self.global_step
        q = 4
        d = self.training_max_steps // q
        n_t = 1 + 8 / (1 + torch.exp(t/self.timesteps))
        r = t * (1 - n_t / q ** (steps // d))
        r = r.int()
        r = torch.clamp(r, min=0, max=self.timesteps-1)
        # sanity check: this should reproduce regular score matching
        #r = torch.zeros_like(t)
        return torch.cat([t, r], dim=-1)

    def load_model_weights_only(self, state_dict):
        """
        Alternative method to load only model weights, bypassing PyTorch Lightning's checkpoint loading.
        Use this instead of trainer.fit() with resume_from_checkpoint for fine-tuning.
        """
        self.load_state_dict(state_dict, strict=False)
        return True

