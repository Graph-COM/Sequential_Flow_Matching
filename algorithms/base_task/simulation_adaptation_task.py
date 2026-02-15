from abc import abstractmethod
from typing import Optional, Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from ..common.abstract_adaptation_task import AbstractAdaptionTask
from .simulation_task import SimulationTask
from utils.logging_utils import energy_score

class SimulationAdaptionTask(AbstractAdaptionTask, SimulationTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg) # suppose to call a Model.init() in the multiple inheritance chain

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        SimulationTask.validation_step(self, batch, batch_idx, namespace=namespace)

    def eval_simulation_sliding_window(self, batch: torch.Tensor, horizon=None, conditions=None,
                                       namespace="validation"):
        """
        Sliding window evaluation for simulation, similar to df_video.py validation_step.
        Uses ground truth context and re-forecasts periodically instead of autoregressive prediction.
        """
        # batch shape: (n_frames, batch_size, observation_dim)
        n_frames, batch_size, *_ = batch.shape

        # Set up sliding window parameters
        prediction_horizon = self.prediction_horizon * self.frame_stack
        n_context_frames = self.context_frames // self.frame_stack
        prediction_horizon = prediction_horizon // self.frame_stack
        num_sliding_windows = n_frames - prediction_horizon - n_context_frames + 1
        open_loop_horizon = self.open_loop_horizon
        # assert open_loop_horizon <= prediction_horizon
        assert open_loop_horizon == 1  # an adaptor model must be closed-loop

        # Initialize prediction storage
        xs_pred_best = torch.zeros_like(batch)
        xs_pred_at_ts = []
        xs_gt_at_ts = []
        mse_list = []

        if self.save_inference:
            if conditions:
                self.save_inference_buffer.append([batch.detach().cpu(), conditions.detach().cpu()])  # initialize the save buffer for current batch
            else:
                self.save_inference_buffer.append([batch.detach().cpu()])  # initialize the save buffer for current batch

        # Sliding window loop
        for step in range(0, num_sliding_windows):
            t_real = n_context_frames + step
            end = min(t_real + prediction_horizon, n_frames)
            prediction_window = [i for i in range(t_real, end)]
            # print(prediction_window)

            # Re-forecast based on new ground-truth observations
            if step % open_loop_horizon == 0:
                # Ground-truth context before t_real is observed
                if step > 0 and self.cfg.update_filtering:
                    xs_pred_best = self.prediction_filtering(xs_pred_best, batch, [i for i in range(t_real)])
                xs_pred_best[:t_real] = batch[:t_real].clone()

                # Generate predictions for the prediction window
                if step == 0 or self.save_inference:
                    xs_pred_best[prediction_window], _ = self.model.model_sampling(
                        # when replanning, use ground-truth context or previous prediction
                        xs_context=batch[:t_real],
                        # xs_context=xs_pred_best[:t_real],
                        conditions=conditions[:t_real + len(prediction_window)] if conditions is not None else None,  # use force before prediction end time
                        prediction_window=prediction_window,
                        n_frames=n_frames,
                        batch_size=batch_size
                    )
                else:
                    xs_pred_best[prediction_window] = self.model.prediction_update(xs_context=batch[:t_real],
                                                                             xs_best_pred=xs_pred_best,
                                                                             conditions=conditions[
                                                                                 :t_real + len(prediction_window)] if conditions is not None else None,
                                                                             prediction_window=prediction_window,
                                                                             n_frames=n_frames,
                                                                             batch_size=batch_size)

            # fetch the best estimation so far
            unnorm_xs_pred = self._unnormalize_x(xs_pred_best[t_real:t_real + prediction_horizon]) * self.rescaler
            unnorm_gt = self._unnormalize_x(batch[t_real:t_real + prediction_horizon]) * self.rescaler
            xs_pred_at_ts.append(unnorm_xs_pred.clone())
            xs_gt_at_ts.append(unnorm_gt)

            # cache the best estimation in the rollout window
            if self.save_inference:
                self.save_inference_buffer[-1].append(
                    [xs_pred_best[prediction_window].detach().cpu(), torch.tensor([t_real]).tile(batch_size).cpu()])

        # xs_pred_at_ts: (num_windows, pred_horizon, batch, obs_dim)
        xs_pred_at_ts = torch.stack(xs_pred_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)

        # TODO: this seems to only record one batch, which is currently fine for 1D burgers (only 100 data points).
        # (1) Global MSE: average over all sliding windows, prediction horizons, batch, obs_dim
        mse = (xs_pred_at_ts - xs_gt_at_ts).square()
        global_mse = mse.mean().item()
        mse_per_horizon = mse.mean(dim=(0, 2, 3))
        mse_per_window = mse.mean(dim=(1, 2, 3))
        global_rmse = np.sqrt(global_mse)
        rmse_per_horizon = mse_per_horizon.sqrt()
        rmse_per_window = mse_per_window.sqrt()

        self.log(f"{namespace}/rmse", global_rmse, on_step=False, on_epoch=True, sync_dist=True)

        # (2) MSE on different prediction horizons: average over sliding windows, batch, obs_dim
        title = "rmse_per_horizon"
        self.log_line_plot(f"plots/{namespace}/{title}", rmse_per_horizon.cpu().numpy(),
                           "prediction_horizon", "rmse", title)

        # # (3) MSE on different sliding windows: average over pred_horizon, batch, obs_dim
        title = "rmse_per_window"
        self.log_line_plot(f"plots/{namespace}/{title}", rmse_per_window.cpu().numpy(),
                           "window_index", "rmse", title)

        if self.cfg.get('num_gen_trials') is not None and self.cfg.get('num_gen_trials') > 1:
            # compute energy score
            num_gen_trials = self.cfg.get('num_gen_trials')
            num_windows, frame, batch_size, x_dim = xs_pred_at_ts.shape
            real_batch_size = batch_size // num_gen_trials
            xs_pred_reshape = xs_pred_at_ts.view([num_windows, frame, real_batch_size, num_gen_trials, x_dim])
            xs_reshape = xs_gt_at_ts.view([num_windows, frame, real_batch_size, num_gen_trials, x_dim])
            xs_reshape = xs_reshape[:, :, :, 0]
            es = []
            for w in range(num_windows):
                es.append(energy_score(xs_pred_reshape[w], xs_reshape[w]))
            es = torch.tensor(es).cpu().numpy()
            title = "es_per_window"
            self.log_line_plot(f"plots/{namespace}/{title}", es,
                               "window_index", "energy_score", title)
            self.log(f"{namespace}/energy_score", es.mean(), on_step=False, on_epoch=True, sync_dist=True)
