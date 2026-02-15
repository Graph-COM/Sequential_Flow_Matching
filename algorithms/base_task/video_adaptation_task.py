from abc import abstractmethod
from typing import Optional, Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torch
from utils.logging_utils import log_video, get_validation_metrics_for_videos, get_validation_metrics_for_stream_videos
import torch.nn.functional as F
from ..common.abstract_adaptation_task import AbstractAdaptionTask
from .video_task import VideoTask

class VideoAdaptionTask(AbstractAdaptionTask, VideoTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg) # suppose to call a Model.init() in the multiple inheritance chain

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        # in case we want to run multiple diffusion trials
        num_trials = self.cfg.get('num_gen_trials')
        if num_trials is not None and num_trials > 1:
            xs = xs.repeat_interleave(num_trials, dim=1)

        n_frames, batch_size, *_ = xs.shape

        # streaming prediction parameters
        prediction_horizon = self.prediction_horizon * self.frame_stack
        n_context_frames = self.context_frames // self.frame_stack
        prediction_horizon = prediction_horizon // self.frame_stack
        num_sliding_windows = n_frames - prediction_horizon - n_context_frames + 1
        open_loop_horizon = self.open_loop_horizon
        # xs_pred = xs[:n_context_frames].clone()

        # best prediction
        xs_best_pred = torch.zeros_like(xs)
        xs_pred_at_ts = []
        xs_gt_at_ts = []
        if self.save_inference:
            self.save_inference_buffer.append([xs.detach().cpu()])  # initialize the save buffer for current batch

        # pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        for step in range(0, num_sliding_windows):
            # actual physical time
            t_real = n_context_frames + step
            prediction_window = [i for i in
                                 range(t_real, min(t_real + prediction_horizon + open_loop_horizon - 1, n_frames))]

            if step % open_loop_horizon == 0:
                if step > 0 and self.cfg.update_filtering:
                    xs_best_pred = self.prediction_filtering(xs_best_pred, xs, [i for i in range(t_real)])
                # ground-truth context before t_real is observed
                xs_best_pred[:t_real] = xs[:t_real].clone()
                # trigger prediction model
                if step == 0 or self.save_inference:
                    # prediction initialization: call slow model
                    # if we are saving inference results (creating offline dataset), always call pretrained slow model
                    xs_best_pred[prediction_window], xs_pred_hist = self.model.model_sampling(xs_context=xs[:t_real],
                                                                                                conditions=conditions,
                                                                                                prediction_window=prediction_window,
                                                                                                n_frames=n_frames,
                                                                                                batch_size=batch_size)
                    # pass
                else:
                    xs_best_pred[prediction_window] = self.model.prediction_update(xs_context=xs[:t_real],
                                                                             xs_best_pred=xs_best_pred,
                                                                             conditions=conditions,
                                                                             prediction_window=prediction_window,
                                                                             n_frames=n_frames,
                                                                             batch_size=batch_size)

            # fetch the best estimation within the prediction horizon so far
            xs_pred_at_ts.append(xs_best_pred[t_real:t_real + prediction_horizon].clone())
            xs_gt_at_ts.append(xs[t_real:t_real + prediction_horizon])

            # cache the best estimation in the rollout window
            if self.save_inference:
                self.save_inference_buffer[-1].append(
                    [xs_best_pred[prediction_window].detach().cpu(), torch.tensor([t_real]).tile(batch_size).detach().cpu()])

        xs_pred_at_ts = torch.stack(xs_pred_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)

        # FIXME: loss
        loss = F.mse_loss(xs_pred_at_ts, xs_gt_at_ts, reduction="mean")
        # loss = F.mse_loss(xs_pred, xs, reduction="none")
        # loss = self.reweight_loss(loss, masks)

        xs_gt_at_ts = self._unstack_and_unnormalize(xs_gt_at_ts)
        xs_pred_at_ts = self._unstack_and_unnormalize(xs_pred_at_ts)

        # cache batch inference results for evaluation
        # self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(),
        # xs_gt_at_ts.detach().cpu(), self._unnormalize_x(xs).detach().cpu()))
        #self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(), xs_gt_at_ts.detach().cpu()))
        self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(),
                                             xs_gt_at_ts.detach().cpu(), self._unnormalize_x(xs).detach().cpu()))

        self.log("validation/loss", loss)

        print('batch validation loss: %.3f ' % loss)

        return loss

