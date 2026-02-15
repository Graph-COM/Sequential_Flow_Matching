from omegaconf import DictConfig
from algorithms.diffusion_forcing.df_video import DiffusionForcingVideo
from algorithms.diffusion_adaptor.df_base_adaptor import DiffusionAdaptor
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
class DiffusionForcingVideoFineTuner(DiffusionAdaptor, DiffusionForcingVideo):
    """
    A fine-tuning version of DiffusionForcingVideo that loads a pretrained checkpoint
    and creates a copy for fine-tuning while keeping the original model intact.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)



    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        # in case we want to run multiple diffusion trials
        num_df = self.cfg.get('num_diffusion_trials')
        if num_df is not None and num_df > 1:
            xs = xs.repeat(1, num_df, 1, 1, 1)

        n_frames, batch_size, *_ = xs.shape

        # streaming prediction parameters
        prediction_horizon = 4 * self.frame_stack  # TODO: load from cfg
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
            self.save_inference_buffer.append([xs]) # initialize the save buffer for current batch

        # pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        for step in range(0, num_sliding_windows):
            # actual physical time
            t_real = n_context_frames + step

            # decide model's rollout length
            if self.update_strategy == 'fixed_horizon':
                # model's rollout horizon is the same as prediction horizon
                rollout_horizon = prediction_horizon + open_loop_horizon - 1
            elif self.update_strategy == 'shrinking_horizon':
                # model rollouts all to the end
                rollout_horizon = n_frames - t_real

            # the prediction window
            prediction_window = [i for i in
                                 range(t_real, min(t_real + rollout_horizon, n_frames))]

            # ground-truth context before t_real is observed
            xs_best_pred[:t_real] = xs[:t_real].clone()

            if step % open_loop_horizon == 0:
                # trigger prediction model
                if step == 0 or self.save_inference:
                    # prediction initialization: call slow model
                    # if we are saving inference results (creating offline dataset), always call pretrained slow model
                    xs_best_pred[prediction_window], xs_pred_hist = self.diffusion_forcing_sampling(xs_context=xs[:t_real],
                                                                                                    conditions=conditions,
                                                                                                    prediction_window=prediction_window,
                                                                                                    n_frames=n_frames,
                                                                                                    batch_size=batch_size)
                    #pass
                else:
                    xs_best_pred[prediction_window] = self.prediction_update(xs_context=xs[:t_real],
                                                                             xs_best_pred = xs_best_pred,
                                                                             conditions=conditions,
                                                                             prediction_window=prediction_window,
                                                                             n_frames=n_frames,
                                                                             batch_size=batch_size)


            # fetch the best estimation within the prediction horizon so far
            xs_pred_at_ts.append(xs_best_pred[t_real:t_real + prediction_horizon].clone())
            xs_gt_at_ts.append(xs[t_real:t_real + prediction_horizon])

            # cache the best estimation in the rollout window
            if self.save_inference:
                self.save_inference_buffer[-1].append([xs_best_pred[prediction_window], torch.tensor([t_real]).tile(batch_size)])


        xs_pred_at_ts = torch.stack(xs_pred_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)

        # FIXME: loss
        loss = F.mse_loss(xs_pred_at_ts, xs_gt_at_ts, reduction="mean")
        # loss = F.mse_loss(xs_pred, xs, reduction="none")
        # loss = self.reweight_loss(loss, masks)

        xs_gt_at_ts = self._unstack_and_unnormalize(xs_gt_at_ts)
        xs_pred_at_ts = self._unstack_and_unnormalize(xs_pred_at_ts)

        # cache batch inference results for evaluation
        #self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(),
        #xs_gt_at_ts.detach().cpu(), self._unnormalize_x(xs).detach().cpu()))
        self.validation_step_outputs.append((xs_pred_at_ts.detach().cpu(),xs_gt_at_ts.detach().cpu()))

        self.log("validation/loss", loss)


        print('batch validation loss: %.3f ' % loss)

        return loss

