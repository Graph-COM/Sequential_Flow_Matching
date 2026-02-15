import os

import numpy as np
from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from einops import rearrange, repeat, reduce
from utils.logging_utils import get_validation_metrics_for_stream_videos
import torch.nn.functional as F
from ..common.abstract_task import AbstractTask
import lpips

class VideoTask(AbstractTask):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.metrics = cfg.metrics
        self.n_tokens = cfg.n_frames // cfg.frame_stack  # number of max tokens for the model
        self.open_loop_horizon = cfg.open_loop_horizon
        self.frame_stack = cfg.frame_stack
        self.context_frames = cfg.context_frames
        self.prediction_horizon = cfg.get('prediction_horizon')
        self.causal = cfg.causal
        self.external_cond_dim = cfg.external_cond_dim

        super().__init__(cfg)


        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        #self.validation_lpips_model = lpips.LPIPS(net='vgg')
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None
        self.validation_step_outputs = []


    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output_dict = super().training_step(batch, batch_idx)
        # log the video
        #if batch_idx % 5000 == 0 and self.logger:
            #log_video(
                #output_dict["xs_pred"],
                #output_dict["xs"],
                #step=self.global_step,
                #namespace="training_vis",
                #logger=self.logger.experiment,
            #)
        return output_dict

    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return
        xs_pred = []
        xs = []
        raw_videos = []
        for pred, gt, raw_video in self.validation_step_outputs:
            xs_pred.append(pred)
            xs.append(gt)
            raw_videos.append(raw_video)
        xs_pred = torch.cat(xs_pred, 2)
        xs = torch.cat(xs, 2)
        #raw_videos = torch.cat(raw_videos, 1)

        if self.logger:
            self.log_stream_video(xs[:, :, 0], xs_pred[:, :, 0])

        #metric_dict = get_validation_metrics_for_videos(
            #xs_pred[self.context_frames :],
            #xs[self.context_frames :],
            #lpips_model=self.validation_lpips_model,
            #fid_model=self.validation_fid_model,
            #fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
        #)
        metric_dict = get_validation_metrics_for_stream_videos(xs_pred, xs, lpips_model=self.validation_lpips_model,
                                    fid_model=self.validation_fid_model,
                                    fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
                                                               num_gen_trials=self.cfg.get('num_gen_trials'))

        # draw
        for title in [k for k in metric_dict.keys() if "_per_" in k]:
            metric, x_label = title.split("_per_")
            self.log_line_plot(f"{namespace}/{title}", metric_dict[title],
                               x_label+'_id', metric, title)

        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items() if "_per_" not in k},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        # in case we want to run multiple diffusion trials
        num_trials = self.cfg.get('num_gen_trials')
        if num_trials is not None and num_trials > 1:
            xs = xs.repeat_interleave(num_trials, dim=1)

        n_frames, batch_size, *_ = xs.shape


        # streaming prediction
        prediction_horizon = self.prediction_horizon * self.frame_stack # TODO: load from cfg
        n_context_frames = self.context_frames // self.frame_stack
        prediction_horizon = prediction_horizon // self.frame_stack
        num_sliding_windows = n_frames - prediction_horizon - n_context_frames + 1
        open_loop_horizon = self.open_loop_horizon
        #xs_pred = xs[:n_context_frames].clone()

        # best prediction
        xs_pred_best = torch.zeros_like(xs)
        xs_pred_at_ts = []
        xs_gt_at_ts = []

        #pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        for step in range(0, num_sliding_windows):
            t_real = n_context_frames + step
            prediction_window = [i for i in
                                 range(t_real, min(t_real + prediction_horizon + open_loop_horizon - 1, n_frames))]
            # xs_pred_all = []
            # re-forecast based on new groud-truth observations
            if step % open_loop_horizon == 0:
                # ground-truth context before t_real is observed
                xs_pred_best[:t_real] = xs[:t_real].clone()

                xs_pred_best[prediction_window], _ = self.model.model_sampling(xs_context=xs[:t_real],
                                                                                                conditions=conditions,
                                                                prediction_window=prediction_window, n_frames=n_frames,
                                                                                                batch_size=batch_size)
            # fetch the best estimation so far
            xs_pred_at_ts.append(xs_pred_best[t_real:t_real+prediction_horizon].clone())
            xs_gt_at_ts.append(xs[t_real:t_real+prediction_horizon])

        xs_pred_at_ts = torch.stack(xs_pred_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)

        loss = F.mse_loss(xs_pred_at_ts, xs_gt_at_ts, reduction="mean")
        print('Validation batch loss: %.3f' % loss)
        #loss = F.mse_loss(xs_pred, xs, reduction="none")
        #loss = self.reweight_loss(loss, masks)

        xs_gt_at_ts = self._unstack_and_unnormalize(xs_gt_at_ts)
        xs_pred_at_ts = self._unstack_and_unnormalize(xs_pred_at_ts)
        self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(),
                                             xs_gt_at_ts.detach().cpu(), self._unnormalize_x(xs).detach().cpu()))

        return loss


