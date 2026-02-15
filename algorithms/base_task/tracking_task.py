from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from ..common.abstract_task import AbstractTask
from omegaconf import OmegaConf
import torch.nn.functional as F
from utils.logging_utils import energy_score

class TrackingTask(AbstractTask):
    def __init__(self, cfg: DictConfig):
        self.observation_dim = cfg.observation_shape[0]
        self.state_dim = cfg.state_dim
        self.n_frames = cfg.n_frames
        self.unstacked_dim = self.state_dim
        cfg.x_shape = (self.unstacked_dim,)
        # self.gamma = cfg.gamma
        # need to manually save these statistics first
        self.observation_mean_path = cfg.observation_mean
        self.observation_std_path = cfg.observation_std
        self.state_mean_path = cfg.state_mean
        self.state_std_path = cfg.state_std
        self._obs_mean = None
        self._obs_std = None
        self._state_mean = None
        self._state_std = None
        #self.rescaler = cfg.rescaler
        #self.open_loop_horizon = cfg.get('open_loop_horizon', 1)
        self.open_loop_horizon = 1 # state tracking is naturally closed-loop
        #mean = list(self.observation_mean)
        #std = list(self.observation_std)

        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.context_frames = 0 # for state tracking, we don't need context
        self.prediction_horizon = 1 # for state tracking, each time we predict current state
        self.chunk_size = 1
        self.external_cond_dim = self.observation_dim
        self.causal = cfg.causal

        self.validation_step_outputs = []



        super().__init__(cfg)

    @property
    def state_mean(self):
        if self._state_mean is None:
            self._state_mean = np.load(self.state_mean_path)[: self.state_dim]
        return self._state_mean
    @property
    def state_std(self):
        if self._state_std is None:
            self._state_std = np.load(self.state_std_path)[: self.state_dim]
        return self._state_std
    @property
    def observation_mean(self):
        if self._obs_mean is None:
            self._obs_mean = np.load(self.observation_mean_path)[: self.observation_dim]
        return self._obs_mean
    @property
    def observation_std(self):
        if self._obs_std is None:
            self._obs_std = np.load(self.observation_std_path)[: self.observation_dim]
        return self._obs_std


    def _preprocess_batch(self, batch):
        # input example: [batch_sizes, type, nframes, observation_dim]
        x = batch[0]  # state [batchsizes, nframes, observation_dim]
        obs = batch[1]

        batch_size, n_frames = x.shape[:2]
        masks = torch.ones(n_frames, batch_size).to(x.device)

        x = self._normalize_x_with_mean(x, self.state_mean, self.state_std)
        obs = self._normalize_x_with_mean(obs, self.observation_mean, self.observation_std)

        conditions = obs # we use observations as our conditions

        x = rearrange(x, "b (t fs) d -> t b (fs d)", fs=self.frame_stack)
        conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack)
        return x, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(2, n_tokens + 1, (batch_size,), device=self.device)
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[:, :n_tokens].bool()
            random_terminal = repeat(random_terminal, "b t -> (t fs) b", fs=self.frame_stack)
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.model(xs, conditions, masks)

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize_with_mean(xs, self.state_mean, self.state_std)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize_with_mean(xs_pred, self.state_mean, self.state_std)[self.frame_stack - 1 :]

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def on_validation_epoch_end(self, namespace='validation') -> None:
        if not self.validation_step_outputs:
            return
        xs_pred = []
        xs = []
        for pred, gt in self.validation_step_outputs:
            xs_pred.append(pred)
            xs.append(gt)
        xs_pred = torch.cat(xs_pred, 2) # [n_frames, prediction_horizon, batch_size, ...]
        xs = torch.cat(xs, 2)
        rmse_per_step = (xs_pred - xs).square().mean(dim=[1,2,3]).sqrt()

        if self.cfg.get('num_gen_trials') is not None and self.cfg.get('num_gen_trials') > 1:
            # compute energy score
            num_gen_trials = self.cfg.get('num_gen_trials')
            frame, _, batch_size, x_dim = xs_pred.shape
            real_batch_size = batch_size // num_gen_trials
            xs_pred_reshape = xs_pred.view([frame, real_batch_size, num_gen_trials, x_dim])
            xs_reshape = xs.view([frame, real_batch_size, num_gen_trials, x_dim])
            xs_reshape = xs_reshape[:, :, 0]
            es = energy_score(xs_pred_reshape, xs_reshape)
            self.log(f"{namespace}/energy_score", es, on_step=False, on_epoch=True, sync_dist=True)
            # compute minimum of N
            min_rmse = (xs_pred_reshape - xs_reshape.unsqueeze(2)).square().mean(dim=[0, 1, 3]).min().sqrt()
            self.log(f"{namespace}/min_rmse", min_rmse, on_step=False, on_epoch=True, sync_dist=True)



        # visualization of one trajectory
        traj_gt = xs[:, 0, 0].cpu().numpy()
        traj_pred = xs_pred[:, 0, 1].cpu().numpy()

        self.log_3d_line_plot('visualization',  traj_pred,'pred_trajectory', traj_gt,
                              'real_trajectory', 'state tracking')

        # TODO: permission deny for some reasons
        title = "rmse_per_step"
        self.log_line_plot(f"{namespace}/{title}", rmse_per_step,
                           "time_step", "rmse", title)
        rmse = rmse_per_step.mean()
        self.log(f"{namespace}/rmse", rmse, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)

        # in case we need multiple runs of generation
        if self.cfg.get('num_gen_trials') is not None and self.cfg.num_gen_trials > 1:
            xs = xs.repeat_interleave(self.cfg.num_gen_trials, dim=1)
            conditions = conditions.repeat_interleave(self.cfg.num_gen_trials, dim=1) if conditions is not None else None

        n_frames, batch_size, *_ = xs.shape
        #if self.guidance_scale == 0:
            #namespace += "_no_guidance_random_walk"
        # best prediction
        xs_pred_best = torch.zeros_like(xs)
        xs_pred_at_ts = []
        xs_gt_at_ts = []

        #pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        for step in range(0, n_frames):
            prediction_window = [step]
            prediction_horizon = len(prediction_window)
            # xs_pred_all = []
            # re-forecast based on new groud-truth observations
            xs_pred_best[prediction_window], _ = self.model.model_sampling(xs_context=xs_pred_best[:step],
                                                                               conditions=conditions[:step+prediction_horizon],
                                                                               prediction_window=prediction_window, n_frames=n_frames,
                                                                               batch_size=batch_size)
            # fetch the best estimation so far
            xs_pred_at_ts.append(xs_pred_best[step:step+prediction_horizon].clone())
            xs_gt_at_ts.append(xs[step:step+prediction_horizon])

        xs_pred_at_ts = torch.stack(xs_pred_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)

        loss = F.mse_loss(xs_pred_at_ts, xs_gt_at_ts, reduction="mean")
        print('Validation batch loss: %.3f' % loss)
        #loss = F.mse_loss(xs_pred, xs, reduction="none")
        #loss = self.reweight_loss(loss, masks)

        xs_gt_at_ts = self._unstack_and_unnormalize_with_mean(xs_gt_at_ts, self.state_mean, self.state_std)
        xs_pred_at_ts = self._unstack_and_unnormalize_with_mean(xs_pred_at_ts, self.state_mean, self.state_std)
        self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(),
                                             xs_gt_at_ts.detach().cpu()))

        return loss
