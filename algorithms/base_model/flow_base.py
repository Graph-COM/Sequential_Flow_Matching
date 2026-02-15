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
from .models.flow import FlowMatching


class FlowMatchingBase(BasePytorchAlgo):
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
        self.numerical_stabilizer = cfg.flow.get('numerical_stabilizer')
        self.n_frames = cfg.get('n_frames') if cfg.get('n_frames') is not None else cfg.get('episode_len') + cfg.frame_stack
        self.n_tokens = self.n_frames // cfg.frame_stack

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.flow.timesteps
        self.sampling_timesteps = cfg.flow.sampling_timesteps
        self.clip_noise = cfg.flow.clip_noise

        self.cfg.flow.cum_snr_decay = self.cfg.flow.cum_snr_decay ** (self.frame_stack * cfg.frame_skip)

        self.validation_step_outputs = []
        super().__init__(cfg)

    def _build_model(self):
        self.flow_model = FlowMatching(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.flow,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = tuple(self.flow_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def forward(self, xs: torch.Tensor, conditions: torch.Tensor, masks: torch.Tensor, context_masks=None) -> torch.Tensor:
        return self.flow_model(xs, conditions, inter_time=self._generate_inter_time(xs, masks=masks,
                                                                                    context_masks=context_masks))

    def model_sampling(self, xs_context, conditions, prediction_window, n_frames, batch_size, diff_hist=False,
                       guidance_fn=None):
        return self.flow_matching_sampling(
            xs_context, conditions, prediction_window, n_frames, batch_size, diff_hist=diff_hist,
            guidance_fn=guidance_fn)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError('Implement training_step in task class')

    def flow_matching_sampling(self, xs_context, conditions, prediction_window, n_frames, batch_size, diff_hist=False,
                               guidance_fn=None):
        xs_pred = xs_context.clone()
        xs_pred_hist = [[] for _ in range(len(prediction_window))] if diff_hist else None
        start_frame = prediction_window[0]
        curr_frame = prediction_window[0]

        pbar = tqdm(total= self.sampling_timesteps * len(prediction_window), initial=0,
                    desc="Flow matching sampling (stacked frame %d to %d)" % (prediction_window[0], prediction_window[-1]))
        while curr_frame <= prediction_window[-1]:
            if self.chunk_size > 0:
                #horizon = min(n_frames - curr_frame, self.chunk_size)
                horizon = min(prediction_window[-1] - curr_frame + 1, self.chunk_size)
            else:
                #horizon = n_frames - curr_frame
                horizon = prediction_window[-1] - curr_frame + 1
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            #start_frame = max(0, curr_frame + horizon - len(pre))

            for m in range(scheduling_matrix.shape[0] - 1):
                from_time = np.concatenate((np.zeros((curr_frame,)), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_time = np.concatenate(
                    (
                        np.zeros((curr_frame,)),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_time = torch.from_numpy(from_time).to(self.device, xs_pred.dtype)
                to_time = torch.from_numpy(to_time).to(self.device, xs_pred.dtype)

                # update xs_pred by naive forward Euler
                xs_pred = self.flow_model.sample_step(xs_pred, conditions[:curr_frame+horizon] if conditions is not None else None,
                                                      from_time, to_time, guidance_fn=guidance_fn)

                pbar.update(horizon)
                pbar.set_postfix(curr_frame=curr_frame, m=m)
            curr_frame += horizon
        return xs_pred[prediction_window].clone(), xs_pred_hist


    def _generate_inter_time(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None,
                             context_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        lower_time, upper_time = self.numerical_stabilizer, 1-self.numerical_stabilizer
        num_frames, batch_size, *_ = xs.shape
        match self.cfg.inter_time:
            case "random_all":  # entirely random noise levels
                dim = (num_frames, batch_size)
                inter_time = self.sample_t(dim, xs.device)
            case "uniform":
                dim = (1, batch_size)
                inter_time = self.sample_t(dim, xs.device)
                inter_time = inter_time.repeat(num_frames, 1)

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            #inter_time = torch.where(discard, torch.full_like(inter_time, self.timesteps - 1), inter_time)
            inter_time = torch.where(discard, torch.full_like(inter_time, 1.0), inter_time)

        if context_masks is not None:
            # for frames that are context, treat as zero noise
            discard = torch.all(~rearrange(context_masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            inter_time = torch.where(discard, torch.zeros_like(inter_time), inter_time)

        inter_time = torch.clip(inter_time, min=lower_time, max=upper_time)
        #inter_time = inter_time * (upper_time - lower_time) + lower_time
        return inter_time

    def _generate_scheduling_matrix(self, horizon: int):
        match self.cfg.scheduling_matrix:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                #return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
                return np.linspace(1-self.numerical_stabilizer, self.numerical_stabilizer, self.sampling_timesteps+1)[:, None].repeat(horizon, axis=1)
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
            #case "trapezoid":
                #return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        #scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        scheduling_matrix = np.zeros((height, horizon))
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, t] /= self.sampling_timesteps

        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
        return np.clip(scheduling_matrix, self.numerical_stabilizer, 1-self.numerical_stabilizer)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def sample_t(self, dim, device):
        if self.cfg.inter_time_dist == 'uniform':
            return torch.rand(dim, device=device)
        elif self.cfg.inter_time_dist == 'log_norm':
            mu, sigma = self.cfg.inter_time_mean, self.cfg.inter_time_std
            normal_samples = torch.randn(dim, device=device) * sigma + mu
            return 1 / (1 + torch.exp(-normal_samples))  # Apply sigmoid
        else:
            raise NotImplementedError('Check your algorithm.inter_time_dist.')