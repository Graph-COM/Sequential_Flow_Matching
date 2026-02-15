import numpy as np
from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from utils.logging_utils import log_video, get_validation_metrics_for_videos, get_validation_metrics_for_stream_videos
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
#from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .abstract_task import AbstractTask
from abc import abstractmethod


class AbstractAdaptionTask(AbstractTask):
    """
    This class is for defining (a) saving offline dataset from a pretrained model; (b) finetuning and (c) inference
    """

    def __init__(self, cfg: DictConfig):
        # Store the original config for reference
        self.original_cfg = cfg
        self.save_inference = cfg.get('save_inference')
        if self.save_inference:
            self.save_inference_buffer = []
        # Fine-tuning configuration
        self.fine_tuning_mode = cfg.get('fine_tuning_mode', True)
        self.update_strategy = cfg.get('update_strategy')
        self.fine_tuning_method = cfg.get('fine_tuning_method')
        self.fine_tuning_teacher_forcing = cfg.get('fine_tuning_teacher_forcing')
        self.train_on_context = cfg.get('train_on_context')
        self.update_chunk_size = cfg.get('update_chunk_size')
        self.update_sampling_timesteps = cfg.get('update_sampling_timesteps')
        self.data_save_dir = cfg.get('data_save_dir')
        self.ckpt_path = cfg.get('ckpt_path')


        # Initialize the parent class first
        super().__init__(cfg)
        
        # Set up full fine-tuning: create a copy of the pretrained model
        #self._setup_full_fine_tuning()

    def load_model_weights_only(self, checkpoint_path):
        """
                Alternative method to load only model weights, bypassing PyTorch Lightning's checkpoint loading.
                Use this instead of trainer.fit() with resume_from_checkpoint for fine-tuning.
                """
        if self.fine_tuning_mode:
            print(f"Loading model weights only from {checkpoint_path}")

            # Load checkpoint manually
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']

            # Create new state dict for fine-tuning structure
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.removeprefix('model.')
                new_state_dict[new_key] = value
            self.model.load_model_weights_only(new_state_dict)
            return True
        return False

    def _preprocess_batch(self, batch):
        return super()._preprocess_batch(batch)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Override the training step for fine-tuning.
        This is where you can implement your custom fine-tuning logic.
        """
        # Preprocess batch to get input_tensor, output_tensor, noise_levels, mask_tensor
        input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, conditions\
                                                    = self._preprocess_batch_fine_tune(batch)
        
        #if self.fine_tuning_method.startswith("predictor"):
            #xs_pred, loss, masks = self.model._fine_tuning_forward(input_tensor, output_tensor, noise_levels, mask_tensor,
                                                        #external_cond=conditions)
        #elif self.fine_tuning_method == 'flow_matching':
        xs_pred, loss, masks = self.model._fine_tuning_flow_forward(input_tensor, output_tensor, noise_levels, flow_time,
                                                        mask_tensor, external_cond=conditions)

        loss = self.reweight_loss(loss, masks)
        # Log the loss using the same pattern as the original code
        print('batch training loss: %.3f ' % loss)
        if batch_idx % 10 == 0:
            self.log("training/loss", loss)

        # For fine-tuning format, use the predictions directly
        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,  # This is the updated_prediction
            "xs": output_tensor,  # Use ground truth for visualization
        }


        return output_dict

    @abstractmethod
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        raise NotImplementedError

    def on_fit_start(self):
        """
        Called when fit begins. Force reset training state for fine-tuning.
        """
        if self.fine_tuning_mode:
            print("Fine-tuning mode: forcing training state reset")
            # Force the trainer to start from the beginning
            if hasattr(self.trainer, 'checkpoint_connector'):
                if hasattr(self.trainer.checkpoint_connector, 'restored_checkpoint_path'):
                    # Clear the restored checkpoint path to force fresh start
                    self.trainer.checkpoint_connector.restored_checkpoint_path = None

    def on_save_checkpoint(self, checkpoint):
        """
        This method is called when a checkpoint is saved.
        You can use this to add custom information to the checkpoint.
        """
        super().on_save_checkpoint(checkpoint)
        
        # Add custom information to the checkpoint
        checkpoint['fine_tuning_mode'] = self.fine_tuning_mode
        checkpoint['original_config'] = self.original_cfg


    def _preprocess_batch_fine_tune(self, batch):
        # preprocess the raw batch = [x_prev_with_context, x_next_with_context, x_gt, pred_time, pad_mask].
        # return input tensor and output tensor for direct training. Depends on which teaching forcing we are doing:
        # (a) teacher forcing by latest estimation: input tensor [new_context, old_prediction],
        # output tensor [new_prediction]
        # (b) teacher forcing by ground truth: input tensor [new_context, old_prediction],
        # output tensor [ground truth]


        # x_prev: [B, max_len, ...], x_next: [B, max_len, ...], x_gt: [B, T, ...], pred_time: [B, 1], pad_mask: [B, max_len]
        x_prev, x_next, x_gt, conditions, pred_time, pad_mask = batch
        pad_mask = pad_mask[0]
        batch_size, context_pred_length = x_prev.shape[:2]

        #if n_frames % self.frame_stack != 0:
        #    raise ValueError("Number of frames must be divisible by frame stack size")
        #if self.context_frames % self.frame_stack != 0:
        #    raise ValueError("Number of context frames must be divisible by frame stack size")

        #n_frames = n_frames // self.frame_stack


        # NOTE: the saved dataset is already in normalized and frame-stacked form
        #x_prev, x_next, x_gt = self._normalize_x(x_prev), self._normalize_x(x_next), self._normalize_x(x_gt)
        #x_prev = rearrange(x_prev, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
        #x_next = rearrange(x_next, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
        #x_gt = rearrange(x_gt, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
        x_prev = rearrange(x_prev, "b t c ... -> t b c ...").contiguous()
        x_next = rearrange(x_next, "b t c ... -> t b c ...").contiguous()
        x_gt = rearrange(x_gt, "b t c ... -> t b c ...").contiguous()
        conditions = rearrange(conditions, "b t c ... -> t b c ...").contiguous() if conditions is not None else None
        pad_mask = rearrange(pad_mask, "b t -> t b").contiguous()

        # TODO: here we assume the context window is of infinite length. Need adaption if considering finite length
        # define input/output tensor
        input_tensor = x_prev
        if self.fine_tuning_teacher_forcing == 'updated_prediction':
            output_tensor = x_next
        elif self.fine_tuning_teacher_forcing.startswith('ground_truth'):
            output_tensor = input_tensor.clone()
            # Vectorized version: create a mask for all time steps that should be replaced
            # the following replacement code essentially does:
            # for b in range(batch_size):
                # output_tensor[pred_time[b]:, b] = x_gt[pred_time[b]:context_pred_length, b]
            # Create time indices for all batch items
            time_indices = torch.arange(context_pred_length, device=pred_time.device).unsqueeze(0).expand(batch_size, -1)  # [batch_size, context_pred_length]
            
            # Create mask: True for time steps >= pred_time[b] for each batch b
            # pred_time is already [batch_size, 1], so we can use it directly
            replacement_mask = time_indices >= pred_time  # [batch_size, context_pred_length]
            
            # Apply the mask to copy from x_gt to output_tensor
            # We need to transpose to work with the [time, batch, ...] format
            replacement_mask = replacement_mask.T
            # Expand mask to match all dimensions after time and batch
            replacement_mask = replacement_mask.view([context_pred_length, batch_size] + [1] * (x_gt.ndim - 2))
            # Vectorized assignment
            output_tensor = torch.where(replacement_mask, x_gt[:context_pred_length], output_tensor)
            if self.fine_tuning_teacher_forcing == 'ground_truth_both':
                replacement_mask[:, (pred_time<=1+self.original_cfg.context_frames)[:, 0], 0] = False # for the first few runs, use model samples
                input_tensor = torch.where(replacement_mask, x_gt[:context_pred_length], input_tensor)
                if self.original_cfg.last_frame_update == 'from_prev_frame':
                    last_frame_ind =  pred_time[:, 0] + self.original_cfg.prediction_horizon - 1
                    input_tensor[last_frame_ind, torch.arange(batch_size)] = input_tensor[last_frame_ind-1, torch.arange(batch_size)].clone()
        # define condition tensor
        # mask out zero padding
        pad_mask_ex = pad_mask.view(*pad_mask.shape, *([1] * (input_tensor.ndim - pad_mask.ndim))).float()
        input_tensor = input_tensor * pad_mask_ex
        output_tensor = output_tensor * pad_mask_ex
        if conditions is not None:
            external_cond_tensor = torch.zeros_like(input_tensor)
            # Vectorized assignment: copy conditions to all batches since all use the same slice
            external_cond_tensor[:context_pred_length] = conditions[:context_pred_length]
            external_cond_tensor = external_cond_tensor * pad_mask_ex
        else:
            external_cond_tensor = None

        # define update_mask tensor that is used for masking the loss, i.e., only =True for entries within update window
        mask_tensor = pad_mask.clone()
        if not self.train_on_context:
            # Vectorized version: create mask for context window
            # it does the following:
            # for b in range(batch_size):
                # mask_tensor[:pred_time[b], b] = False
            # Create time indices for all batch items
            time_indices = torch.arange(context_pred_length, device=pred_time.device).unsqueeze(0).expand(batch_size, -1)  # [batch_size, context_pred_length]
            # Create mask: True for time steps >= pred_time[b] (keep prediction window, mask context)
            keep_mask = time_indices >= pred_time  # [batch_size, context_pred_length]
            # Transpose to work with [time, batch] format and apply mask
            keep_mask_t = keep_mask.T  # [context_pred_length, batch_size]
            mask_tensor[:context_pred_length] = mask_tensor[:context_pred_length] & keep_mask_t


        # get re-noise levels (how much noise added to input tensor) and flow time (the number of generation steps)
        noise_levels, flow_time = self.model._get_renoise_levels_for_batched_input(input_tensor, pred_time, pad_mask)

        return input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, external_cond_tensor


    def save_inference_data(self, save_buffer):
        import os
        import os.path as osp
        if not osp.exists(self.data_save_dir + '/fine-tuning'):
            os.makedirs(self.data_save_dir + '/fine-tuning')
        # TODO: to save space, cast to float16 and use np.savez_compressed
        algorithm_name = self.cfg._name + '_' + self.update_strategy
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

    def on_validation_epoch_end(self, namespace="validation") -> None:
        # TODO: save multiple files
        if self.save_inference:
            # save the inference results into disk for future training of prediction adaption
            self.save_inference_data(self.save_inference_buffer)

        # this should call the base model class's method
        super().on_validation_epoch_end()

    def prediction_filtering(self, xs_best_pred, xs, time_ref):
        # xs_best_pred: [time, batch*num_samples, ...]
        # xs: [time, batch, ...]
        # time_ref: a list of integer
        # compare xs_best_pred[time_ref] and xs[time_ref] and filter the trajectory with lowest error
        num_gen_trials = self.cfg.get('num_gen_trials')
        frame, batch_size, x_dim = xs_best_pred.shape[0], xs_best_pred.shape[1], xs_best_pred.shape[2:]
        real_batch_size = batch_size // num_gen_trials
        new_shape = (frame, real_batch_size, num_gen_trials) + x_dim
        xs_best_reshape = xs_best_pred.view(new_shape)
        xs_best_ref = xs_best_reshape[time_ref]
        xs_ref = xs.view(new_shape)[time_ref]
        dims_to_avg = [i for i in range(xs_best_ref.ndim) if i != 1 and i != 2] # average over all axis except batch and num_samples
        error = (xs_best_ref - xs_ref).square().mean(dim=dims_to_avg)
        # TODO: maybe filter the top-K instead of the best
        filtering_indices = error.argmin(dim=1)
        xs_best_filtered = xs_best_reshape[:, torch.arange(real_batch_size), filtering_indices, :]
        xs_best_filtered = xs_best_filtered.repeat_interleave(num_gen_trials, dim=1)
        return xs_best_filtered