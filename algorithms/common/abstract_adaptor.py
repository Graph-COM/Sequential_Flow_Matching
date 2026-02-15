import numpy as np
from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from utils.logging_utils import log_video, get_validation_metrics_for_videos, get_validation_metrics_for_stream_videos
from algorithms.base_model.models.utils import extract
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
from .base_pytorch_algo import BasePytorchAlgo
from abc import abstractmethod


class AbstractAdaptor(BasePytorchAlgo):
    """
    This class is an abstract Mixin class that defines the finetuning model on top of base model
    """

    def __init__(self, cfg: DictConfig):
        # Store the original config for reference
        self.original_cfg = cfg
        # Fine-tuning configuration
        self.fine_tuning_mode = cfg.get('fine_tuning_mode', True)
        self.update_strategy = cfg.get('update_strategy')
        self.fine_tuning_method = cfg.get('fine_tuning_method')
        self.fine_tuning_teacher_forcing = cfg.get('fine_tuning_teacher_forcing')
        self.train_on_context = cfg.get('train_on_context')
        self.finetune_external_cond_dim = cfg.get('finetune_external_cond_dim')
        self.update_chunk_size = cfg.get('update_chunk_size')
        self.update_sampling_timesteps = cfg.get('update_sampling_timesteps')
        self.data_save_dir = cfg.get('data_save_dir')
        self.ckpt_path = cfg.get('ckpt_path')

        # Initialize the parent class first
        super().__init__(cfg)

        # Set up full fine-tuning: create a copy of the pretrained model
        self._setup_full_fine_tuning()

    def prediction_update(self, xs_context, xs_best_pred, conditions, prediction_window, n_frames, batch_size,
                          guidance_fn=None, **kwargs):
        # prediction update: call fast model
        # update_window = prediction_window if self.update_strategy == 'shrinking_horizon' else prediction_window[:-1]
        update_window = prediction_window
        if self.fine_tuning_method == 'flow_matching':
            xs_best_pred_updated = self.model_sample_update(xs_context=xs_context,
                                                                  xs_best_pred=xs_best_pred,
                                                                  conditions=conditions,
                                                                  update_window=update_window,
                                                                  n_frames=n_frames,
                                                                  batch_size=batch_size, guidance_fn=guidance_fn, **kwargs)
        elif self.fine_tuning_method == 'closed_loop':
            xs_best_pred_updated = self.model_closed_loop_update(xs_context=xs_context,
                                                                  xs_best_pred=xs_best_pred,
                                                                  conditions=conditions,
                                                                  update_window=update_window,
                                                                  n_frames=n_frames,
                                                                  batch_size=batch_size, guidance_fn=guidance_fn, **kwargs)
        return xs_best_pred_updated

    @abstractmethod
    def _setup_full_fine_tuning(self):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def _create_model_copy(self):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def _fine_tuning_flow_forward(self, input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, external_cond=None):
        raise NotImplementedError

    @abstractmethod
    def _fine_tuning_forward(self, input_tensor, output_tensor, noise_levels, mask_tensor, external_cond=None):
        """
        Direct fine-tuning forward pass using the pretrained diffusion model.
        
        Args:
            input_tensor: [H-1+max_context, (H-1)*B, 3, 128, 128] - input sequences
            output_tensor: [H-1+max_context, (H-1)*B, 3, 128, 128] - target sequences
            noise_levels: [H-1+max_context, (H-1)*B] - noise levels for each position
            mask_tensor: [H-1+max_context, (H-1)*B] - mask indicating prediction targets
            
        Returns:
            Tuple of (predictions, loss)
        """
        raise NotImplementedError

    @abstractmethod
    def _get_renoise_levels_for_batched_input(self, input_tensor, pred_time, pad_mask):
        """
        Generate re-noise levels for fine-tuning based on input tensor and configuration.
        For flow matching finetuning, the renoise levels refer to how much noises we will add to the old prediction
        For a seq2seq predictor, the renoise levels simply refer to indices of lead time
        
        Args:
            input_tensor: Input tensor with shape [context_update_length, B, 3, 128, 128]
            pred_time: Physical time when making prediction with shape [B, 1]
        Returns:
            noise_levels: Re-noise levels tensor with shape [context_update_length, B]. This determines the start point
            of the flow
            flow_time: maximal number of flow steps with shape [context_update_length, B]. This determines length of the
            flow path
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        """
        Configure optimizer for direct fine-tuning.
        Train all parameters of the diffusion model.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model_weights_only(self, checkpoint_path):
        """
        Alternative method to load only model weights, bypassing PyTorch Lightning's checkpoint loading.
        Use this instead of trainer.fit() with resume_from_checkpoint for fine-tuning.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_interpolation_flow_time(self, xs: torch.Tensor, flow_time: torch.Tensor,
                                            ) -> torch.Tensor:
        """
        Generate flow matching time for training
        """
        raise NotImplementedError

    @abstractmethod
    def model_sample_update(self, xs_context, xs_best_pred, conditions, update_window, n_frames, batch_size,
                            guidance_fn=None):
        raise NotImplementedError

    @abstractmethod
    def model_closed_loop_update(self, xs_context, xs_best_pred, conditions, update_window, n_frames, batch_size,
                                 guidance_fn=None):
        # this is for CL-diffusion baseline
        raise NotImplementedError

    @abstractmethod
    def _generate_update_scheduling_matrix(self, flow_time):
        raise NotImplementedError

    @staticmethod
    def vectorized_update(target_tensor, pred_time, values):
        """
        Updates a target tensor of shape [T, batch_size] using vectorized indexing.
        """
        T, batch_size = target_tensor.shape
        prediction_horizon = values.shape[0]
        device = target_tensor.device

        # Create a grid of time indices [prediction_horizon, batch_size]
        time_indices = torch.arange(prediction_horizon, device=device).unsqueeze(1) + pred_time.unsqueeze(0)

        # Create a corresponding grid of batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(prediction_horizon, -1)

        # Expand values to match the grid shape
        expanded_values = values.unsqueeze(1).expand(-1, batch_size)

        # Create a mask to prevent writing out-of-bounds
        valid_mask = (time_indices >= 0) & (time_indices < T)

        # Use the mask to get flat, valid indices and values
        flat_time_idx = time_indices[valid_mask]
        flat_batch_idx = batch_indices[valid_mask]
        flat_values = expanded_values[valid_mask]

        # Assign values using advanced indexing: target[time, batch]
        target_tensor[flat_time_idx, flat_batch_idx] = flat_values
        return target_tensor
