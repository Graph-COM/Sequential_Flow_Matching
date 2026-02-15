from abc import abstractmethod
from typing import Optional, Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torch
from utils.logging_utils import log_video, get_validation_metrics_for_videos, get_validation_metrics_for_stream_videos
import torch.nn.functional as F
from ..common.abstract_adaptation_task import AbstractAdaptionTask
from .tracking_task import TrackingTask

class TrackingAdaptionTask(AbstractAdaptionTask, TrackingTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg) # suppose to call a Model.init() in the multiple inheritance chain

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
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


        if self.save_inference:
            self.save_inference_buffer.append([xs.detach().cpu(), conditions.detach().cpu()])  # initialize the save buffer for current batch

        #pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        for step in range(0, n_frames):
            prediction_window = [step]
            prediction_horizon = len(prediction_window)
            # xs_pred_all = []
            # re-forecast based on new groud-truth observations
            if step == 0 or self.save_inference:
                xs_pred_best[prediction_window], _ = self.model.model_sampling(xs_context=xs_pred_best[:step],
                                                                           conditions=conditions[:step+prediction_horizon],
                                                                           prediction_window=prediction_window, n_frames=n_frames,
                                                                           batch_size=batch_size)
            else:
                xs_pred_best[prediction_window] = self.model.prediction_update(xs_context=xs_pred_best[:step],
                                                                           xs_best_pred=xs_pred_best,
                                                                           conditions=conditions[
                                                                               :step + prediction_horizon],
                                                                           prediction_window=prediction_window,
                                                                           n_frames=n_frames,
                                                                           batch_size=batch_size)
            # fetch the best estimation so far
            xs_pred_at_ts.append(xs_pred_best[step:step+prediction_horizon].clone())
            xs_gt_at_ts.append(xs[step:step+prediction_horizon])


            if self.save_inference:
                self.save_inference_buffer[-1].append(
                    [xs_pred_best[prediction_window].detach().cpu(), torch.tensor([step]).tile(batch_size).detach().cpu()])

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


    def save_inference_data(self, save_buffer):
        import os
        import os.path as osp
        if not osp.exists(self.data_save_dir + '/fine-tuning'):
            os.makedirs(self.data_save_dir + '/fine-tuning')
        # TODO: to save space, cast to float16 and use np.savez_compressed
        algorithm_name = self.cfg._name + '_' + self.update_strategy + '_rq%d%d' % (self.original_cfg.r, self.original_cfg.q)
        save_path = self.data_save_dir + '/fine-tuning/%s.pt' % algorithm_name
        unzipped = list(zip(*save_buffer))
        x_gt = torch.cat(unzipped[0], dim=1)
        if isinstance(unzipped[1][0], torch.Tensor):
            conditions = torch.cat(unzipped[1], dim=1)
            pred_start_id = 2
        else:
            conditions = None
            pred_start_id = 1
        x_pred = []
        t_real = []
        print('update_strategy: ', self.update_strategy)
        print('model_wandb_id: ', self.ckpt_path)
        for group in unzipped[pred_start_id:]:
            # group is like ([x_i, t_i] from batch1, [x_i, t_i] from batch2, ...)
            xs, ts = zip(*group)
            xs_all = torch.cat(xs, dim=1)
            ts_all = torch.cat(ts, dim=0)
            x_pred.append(xs_all)
            t_real.append(ts_all)

        torch.save({'ground_truth': x_gt, 'conditions': conditions, 'model_pred': x_pred, 'pred_time': t_real,
                    'frame_stack': self.frame_stack, 'open_loop_horizon': self.open_loop_horizon,
                    'context_frames': self.context_frames, 'update_strategy': self.update_strategy,
                    'model_wandb_id': self.ckpt_path},
                   save_path)
