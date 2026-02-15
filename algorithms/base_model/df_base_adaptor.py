import numpy as np
from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from utils.logging_utils import log_video, get_validation_metrics_for_videos, get_validation_metrics_for_stream_videos
from algorithms.base_model.models.utils import extract
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
from ..common.abstract_adaptor import AbstractAdaptor
from .df_base import DiffusionForcingBase
from omegaconf.listconfig import ListConfig

class DiffusionAdaptor(AbstractAdaptor, DiffusionForcingBase):
    """
    This class is for defining (a) saving offline dataset from a pretrained diffusion model; (b) finetuning and (c) inference
    """

    def __init__(self, cfg: DictConfig):
        # Initialize the parent class first
        super().__init__(cfg)

    def _setup_full_fine_tuning(self):
        """
        Set up full fine-tuning by creating a copy of the pretrained model.
        This ensures we have both the original and fine-tuned models.
        """
        # Store the original pretrained model (frozen) - no need to reassign
        # self.diffusion_model already contains the frozen pretrained model

        # Create a copy for fine-tuning
        self.finetuned_diffusion_model = self._create_model_copy()

        print("Full fine-tuning setup complete:")
        print(f"  - Frozen model: {type(self.diffusion_model).__name__}")
        print(f"  - Fine-tuned model: {type(self.finetuned_diffusion_model).__name__}")
        #print(f"  - Both models have identical architecture and initial weights")
        print(f"  - Both models are initialized with random weights.")

    def _create_model_copy(self):
        """
        Create a deep copy of the diffusion model for fine-tuning.
        
        Returns:
            A copy of the diffusion model with identical weights
        """
        # Import the same model class
        from algorithms.base_model.models.diffusion import Diffusion
        
        # Create a new instance with the same configuration
        finetune_extra_dim = self.finetune_external_cond_dim if self.finetune_external_cond_dim else 0
        # Handle external_cond_dim: it can be a list [C, H, W] or an integer
        original_external_cond_dim = self.diffusion_model.external_cond_dim
        if isinstance(original_external_cond_dim, (list, tuple, ListConfig)):
            # If it's a list, only modify the channel dimension (first element)
            new_external_cond_dim = [original_external_cond_dim[0] + finetune_extra_dim] + list(original_external_cond_dim[1:])
        elif original_external_cond_dim == 0 or original_external_cond_dim is None:
            # If it's 0 or None, just use the finetune_extra_dim
            new_external_cond_dim = finetune_extra_dim if finetune_extra_dim > 0 else 0
        else:
            # If it's an integer, add to it
            new_external_cond_dim = original_external_cond_dim + finetune_extra_dim
        
        model_copy = Diffusion(
            x_shape=self.diffusion_model.x_shape,
            external_cond_dim=new_external_cond_dim,
            is_causal=self.diffusion_model.is_causal,
            cfg=self.diffusion_model.cfg,
        )
        
        # Copy the weights from the original model
        #model_copy.load_state_dict(self.diffusion_model.state_dict())
        
        # Ensure the copy is on the same device
        if hasattr(self.diffusion_model, 'device'):
            model_copy = model_copy.to(self.diffusion_model.device)
        
        #print(f"Created model copy with {sum(p.numel() for p in model_copy.parameters())} parameters")
        
        return model_copy


    def _fine_tuning_flow_forward(self, input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, external_cond=None, guidance_fn=None):
        # Step 1: compute effective "noise" based on noise level (here noise level mean how much noise we added to prev predictions)
        # effective noise: eps_tilde = eps + (sqrt_alphas_cumprod)/(sqrt_one_minus_alphas_cumprod) * (input-output)
        sqrt_alphas_cumprod_at_noise_levels = extract(self.diffusion_model.sqrt_alphas_cumprod, noise_levels, input_tensor.shape)
        sqrt_alphas_cumprod_at_flow_time = extract(self.diffusion_model.sqrt_alphas_cumprod, flow_time, input_tensor.shape)
        sqrt_one_minus_alphas_cumprod_at_noise_levels = extract(self.diffusion_model.sqrt_one_minus_alphas_cumprod, noise_levels,
                                                input_tensor.shape)
        sqrt_one_minus_alphas_cumprod_at_flow_time = extract(self.diffusion_model.sqrt_one_minus_alphas_cumprod, flow_time,
                                                input_tensor.shape)

        residual = sqrt_alphas_cumprod_at_noise_levels * input_tensor - sqrt_alphas_cumprod_at_flow_time * output_tensor
        eps = torch.randn_like(output_tensor)
        eps = torch.clamp(eps, -self.clip_noise, self.clip_noise)
        eps = sqrt_one_minus_alphas_cumprod_at_noise_levels * eps
        eps = eps + residual
        eps = eps / sqrt_one_minus_alphas_cumprod_at_flow_time
        mask_tensor = mask_tensor.view(*mask_tensor.shape, *([1] * (eps.ndim - mask_tensor.ndim)))
        eps = eps * mask_tensor # mask out noises to zero padding part


        # Step 2: sample random flow time
        inter_time = self._generate_interpolation_flow_time(output_tensor, flow_time)


        # Step 3: finetune diffusion model's denoising network to match the effective noise
        xs_pred, loss = self.finetuned_diffusion_model.forward_with_given_noise(
                        output_tensor, external_cond=external_cond, noise_levels=inter_time, noise=eps,
                        guidance_fn=guidance_fn, pad_mask=~mask_tensor[..., 0].T)

        return xs_pred, loss, mask_tensor


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
        # The diffusion model expects [seq_len, batch_size, ...] format, which is already correct!
        # No need to permute input_tensor - it's already in the right format
        
        # Ground-truth context stabilization
        # Convert noise level -1 to stabilization_level - 1
        # clipped_noise_levels = torch.where(
            #noise_levels < 0,
            #torch.full_like(noise_levels, self.finetuned_diffusion_model.stabilization_level - 1, dtype=torch.long),
            #noise_levels,
        #)

        # Treating as stabilization would require us to scale with sqrt of alpha_cum
        #orig_input_tensor = input_tensor.clone().detach()
        #scaled_context = self.finetuned_diffusion_model.q_sample(
            #input_tensor,
            #clipped_noise_levels,
            #noise=torch.zeros_like(input_tensor),
        #)
        #input_tensor = torch.where(
            #self.finetuned_diffusion_model.add_shape_channels(noise_levels < 0),
            #scaled_context,
            #orig_input_tensor
        #)

        # Get model prediction based on the fine-tuned diffusion model
        if self.fine_tuning_method == 'predictor_denoising':
            # use E(x_0|x_t) estimation
            model_pred = self.finetuned_diffusion_model.model_predictions(input_tensor, noise_levels, external_cond=external_cond)
            model_pred = model_pred.pred_x_start  # [seq_len, batch_size, C, H, W]

        elif self.fine_tuning_method == 'predictor_residual':
            # directly use model output
            model_pred = self.finetuned_diffusion_model.model(input_tensor, noise_levels, None,
                                                              is_causal=self.finetuned_diffusion_model.is_causal,
                                                              external_cond=external_cond)
            model_pred = model_pred + input_tensor

        # Apply mask for loss calculation
        mask_expanded = mask_tensor.view(*mask_tensor.shape, *([1] * (model_pred.ndim - mask_tensor.ndim)))

        # Calculate loss only on masked positions
        loss = F.mse_loss(
            model_pred,
            output_tensor,
            reduction='none',
        )
        

        return model_pred, loss, mask_expanded

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
        context_pred_length, batch_size = input_tensor.shape[0], input_tensor.shape[1]

        # Initialize tensors
        batched_noise_levels = torch.zeros(context_pred_length, batch_size, device=input_tensor.device, dtype=torch.int32)
        batched_flow_time = torch.zeros(context_pred_length, batch_size, device=input_tensor.device, dtype=torch.int32)

        # Calculate context lengths and update horizons for all batches
        context_lengths = pred_time.squeeze()  # [batch_size]
        #update_horizons = context_pred_length - context_lengths  # [batch_size]

        # Find unique update horizons
        #unique_horizons, inverse_indices = torch.unique(update_horizons, return_inverse=True)

        # Handle context part
        if self.train_on_context and self.fine_tuning_method == 'flow_matching':
            time_indices = torch.arange(context_pred_length, device=input_tensor.device).unsqueeze(0).expand(batch_size,
                                                                                                             -1)
            context_mask = time_indices < context_lengths.unsqueeze(1)
            context_mask_t = context_mask.T

            context_renoise = (self.timesteps - 1)
            context_flow_time = (self.timesteps - 1)
            batched_noise_levels = torch.where(context_mask_t, context_renoise, batched_noise_levels)
            batched_flow_time = torch.where(context_mask_t, context_flow_time, batched_flow_time)

        # Vectorized assignment for renoise levels and flow time
        prediction_horizon = self.cfg.prediction_horizon // self.frame_stack
        renoise_levels, flow_time = self._generate_renoise_level_and_flow_time(
            update_horizon=prediction_horizon, device=input_tensor.device)
        batched_noise_levels = self.vectorized_update(batched_noise_levels, context_lengths, renoise_levels)
        batched_flow_time = self.vectorized_update(batched_flow_time, context_lengths, flow_time)

        # Mask out zero padding
        batched_noise_levels = batched_noise_levels * pad_mask.int()
        batched_flow_time = batched_flow_time * pad_mask.int()

        return batched_noise_levels, batched_flow_time

    def model_sample_update(self, xs_context, xs_best_pred, conditions, update_window, n_frames, batch_size,
                          guidance_fn=None, **kwargs):
        xs_pred = xs_context.clone()
        start_frame = update_window[0]
        end_frame = update_window[-1]
        curr_frame = update_window[0]

        renoise_levels, flow_time = self._generate_renoise_level_and_flow_time(len(update_window), xs_pred.device)
        # handle last frame that we do not have old prediction of
        # if we are using shrinking horizon or update from noise, then no need to handle it
        if self.update_strategy == 'fixed_horizon' and self.cfg.last_frame_update == 'from_prev_frame':
            if 'last_frame_pad' not in kwargs or kwargs.get('last_frame_pad'): # handle hybrid strategy
                xs_best_pred[end_frame] = xs_best_pred[end_frame-1]

        while curr_frame <= end_frame:
            update_idx = curr_frame - start_frame
            if self.update_chunk_size > 0:
                horizon = min(end_frame - curr_frame + 1, self.update_chunk_size)
            else:
                horizon = end_frame - curr_frame + 1
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."

            # starting with small noisy version of old prediction
            xs_to_update = xs_best_pred[curr_frame:curr_frame + horizon]
            chunk = self.diffusion_model.q_sample(xs_to_update, renoise_levels[update_idx:update_idx+horizon, None].tile((1, batch_size)))
            if (renoise_levels == 999).all():
                # fully noising case
                chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
                chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            #start_frame = max(0, curr_frame + horizon - len(pre))

            # scheduling matrix
            scheduling_matrix = self._generate_update_scheduling_matrix(flow_time[update_idx:update_idx+horizon].cpu().numpy())

            pbar = tqdm(total= scheduling_matrix.shape[0]-1, initial=0,
                    desc="Diffusion samples updating (stacked frame %d to %d)" % (curr_frame, curr_frame+horizon-1))

            for m in range(scheduling_matrix.shape[0] - 1):
                # TODO: add zero paddings for maze planning task
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

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
                xs_pred = self.finetuned_diffusion_model.sample_step(xs_pred, conditions[:curr_frame+horizon] if conditions is not None else None,
                                                           from_noise_levels, to_noise_levels, guidance_fn=guidance_fn)

                pbar.update(horizon)
                pbar.set_postfix(m=m)
            curr_frame += horizon
        return xs_pred[update_window].clone()


    def configure_optimizers(self):
        """
        Configure optimizer for direct fine-tuning.
        Train all parameters of the diffusion model.
        """
        if self.fine_tuning_mode:
            # Train all parameters of the diffusion model
            optimizer = torch.optim.AdamW(
                self.finetuned_diffusion_model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.optimizer_beta
            )

            print("Configured optimizer for direct fine-tuning of diffusion model")
            return optimizer
        else:
            # Use the parent's optimizer configuration
            return super().configure_optimizers()

    def load_model_weights_only(self, state_dict):
        """
        Alternative method to load only model weights, bypassing PyTorch Lightning's checkpoint loading.
        Use this instead of trainer.fit() with resume_from_checkpoint for fine-tuning.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('diffusion_model.'):
                # Load into frozen model (self.flow_model)
                new_state_dict[key] = value
                # Load into fine-tuned model (self.finetuned_flow_model)
                finetuned_key = key.replace('diffusion_model.', 'finetuned_diffusion_model.')
                new_state_dict[finetuned_key] = value
            else:
                # Keep other keys as they are
                new_state_dict[key] = value
            # Load the state dict
        self.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully: identical weights for pretrained and finetuned models")
        return True

    def load_heterog_model_weights_only(self, state_dict):
        """
        Alternative method to load only model weights, bypassing PyTorch Lightning's checkpoint loading.
        Use this instead of trainer.fit() with resume_from_checkpoint for fine-tuning.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('diffusion_model.'):
                # Load into frozen model (self.flow_model)
                new_state_dict[key] = value
                # Load into fine-tuned model (self.finetuned_flow_model)
                finetuned_key = key.replace('diffusion_model.', 'finetuned_diffusion_model.')
                # init_mlp could be of different dim in pretrained and finetuned model
                if 'init_mlp' in finetuned_key:
                    new_value = self.state_dict()[finetuned_key]
                    if new_value.ndim == 2:
                        new_value[:, :value.size(1)] = value
                    else:
                        new_value[:value.size(0)] = value
                    new_state_dict[finetuned_key] = new_value
                else:
                    new_state_dict[finetuned_key] = value
            else:
                # Keep other keys as they are
                new_state_dict[key] = value
            # Load the state dict
        self.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully: heterogeneous weights for pretrained and finetuned models")
        return True

    def _generate_interpolation_flow_time(self, xs: torch.Tensor, flow_time: torch.Tensor,
                                          ) -> torch.Tensor:
        """
        Generate flow matching time for training
        """
        num_frames, batch_size, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_all":  # entirely random noise levels
                dim = (num_frames, batch_size)
                noise_levels = self.sample_t(dim, xs.device)
            case "uniform":
                dim = (1, batch_size)
                noise_levels = self.sample_t(dim, xs.device)
                noise_levels = noise_levels.repeat(num_frames, 1)

        noise_levels = (noise_levels * flow_time).long() # scale [0, 1] to desired [0, flow_time]
        return noise_levels


    def _generate_update_scheduling_matrix(self, flow_time):
        #flow_time = np.rint(flow_time * (self.cfg.diffusion.sampling_timesteps / self.cfg.diffusion.timesteps)).astype(np.int32)
        match self.cfg.get('update_scheduling_matrix'):
            #case 'pyramid':
                #return self._generate_update_pyramid_scheduling_matrix(flow_time, self.uncertainty_scale)
            #case 'full_sequence':
                #return self._generate_update_pyramid_scheduling_matrix(flow_time, uncertainty_scale=0)
            #case 'autoregressive':
                #return self._generate_update_autoregressive_scheduling_matrix(flow_time)
            case 'shortest':
                return self._generate_update_shortest_scheduling_matrix(flow_time)

    def _generate_update_pyramid_scheduling_matrix(self, flow_time, uncertainty_scale: float):
        horizon = len(flow_time)
        height = flow_time.max() + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int32)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = flow_time[t] + int(t * uncertainty_scale) - m
                scheduling_matrix[m, t] = np.clip(scheduling_matrix[m, t], 0, flow_time[t])

        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
        return scheduling_matrix

    def _generate_update_autoregressive_scheduling_matrix(self, flow_time):
        horizon = len(flow_time)
        flow_time_cumsum = np.cumsum(flow_time)
        height = sum(flow_time) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = flow_time_cumsum[t] - m
                scheduling_matrix[m, t] = np.clip(scheduling_matrix[m, t], 0, flow_time[t])

        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
        return scheduling_matrix

    def _generate_update_shortest_scheduling_matrix(self, flow_time):
        horizon = len(flow_time)
        height = self.update_sampling_timesteps + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int32)
        for t in range(horizon):
            schedule_at_t = np.linspace(flow_time[t], 0, height)
            schedule_at_t = np.rint(schedule_at_t).astype(np.int32)
            scheduling_matrix[:, t] = schedule_at_t

        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
        return scheduling_matrix

    def _generate_renoise_level_and_flow_time(self, update_horizon, device='cpu'):
        # get renoise level and flow time
        if self.cfg.renoise_factor_of_lead_time == 'linear':
            renoise_factor_of_lead_time = torch.arange(1, update_horizon + 1,
                                                       device=device)
        elif self.cfg.renoise_factor_of_lead_time == 'constant':
            renoise_factor_of_lead_time = torch.ones(update_horizon, device=device).int()
        if self.cfg.flow_time_factor_of_lead_time == 'linear':
            flow_time_factor_of_lead_time = torch.arange(1, update_horizon + 1,
                                                         device=device)
        elif self.cfg.flow_time_factor_of_lead_time == 'constant':
            flow_time_factor_of_lead_time = torch.ones(update_horizon, device=device).int()
        renoise_levels = (self.cfg.renoise_level * renoise_factor_of_lead_time)
        flow_time = (self.cfg.flow_time * flow_time_factor_of_lead_time)

        # if predict last frame from noise, then set renoise level and flow time to 1
        if self.cfg.update_strategy == 'fixed_horizon' and self.cfg.last_frame_update == 'from_noise':
            renoise_levels[-1], flow_time[-1] = self.diffusion_model.timesteps - 1, self.diffusion_model.timesteps - 1

        return renoise_levels, flow_time

    def sample_t(self, dim, device):
        # sample from [0, 1]
        if self.cfg.noise_level_dist == 'uniform':
            return torch.rand(dim, device=device)
        elif self.cfg.noise_level_dist == 'log_norm':
            mu, sigma = self.cfg.noise_level_mean, self.cfg.noise_level_std
            normal_samples = torch.randn(dim, device=device) * sigma + mu
            return 1 / (1 + torch.exp(-normal_samples))  # Apply sigmoid


    def model_closed_loop_update(self, xs_context, xs_best_pred, conditions, update_window, n_frames, batch_size,
                                guidance_fn=None, **kwargs):
        start_frame = update_window[0]
        end_frame = update_window[-1]
        curr_frame = update_window[0]
        prediction_horizon = len(update_window)
        # Use global_step for control tasks.
        global_step = kwargs.get('steps', None)

        noise_levels = torch.tensor([(i / prediction_horizon) * (self.timesteps-1) for i in range(1, prediction_horizon + 1)]).to(
            xs_context.device)
        #noise_levels[...] = 999 # sanity check
        noise_levels = noise_levels.int()
        # if this is the very first update step, renoise the previous prediction.
        if (update_window[0] == (self.cfg.context_frames // self.cfg.frame_stack) + 1 and global_step is None) or global_step == 1:
            xs_pred = self.diffusion_model.q_sample(xs_best_pred[update_window[:-1]],
                                               noise_levels[:-1][:, None].tile((1, batch_size)))
        else:
            # there are two modes of closed-loop diffusion: (1) clean: always keep clean estimation, so we need to renoise and denoise;
            # (2) noisy: we maintain partially noisy states, so we only need to denoise
            if self.cfg.update_scheduling_matrix == 'clean':
                xs_pred = self.diffusion_model.q_sample(xs_best_pred[update_window[:-1]],
                                                        noise_levels[:-1][:, None].tile((1, batch_size)))
            else:
                xs_pred = xs_best_pred[update_window[:-1]].clone()

        # append a Gaussian noise to the end
        xs_pred = torch.cat([xs_pred, torch.randn_like(xs_pred)[0:1]], dim=0)
        # add context
        xs_pred = torch.cat([xs_context, xs_pred], dim=0)

        # partially denoise
        from_time_step = torch.cat([torch.zeros((curr_frame,), dtype=torch.int64).to(xs_pred.device), noise_levels])[
            :, None
        ].tile([1, batch_size])
        to_time_step = torch.cat([torch.zeros((curr_frame + 1,), dtype=torch.int64).to(xs_pred.device), noise_levels[:-1]])[
            :, None
        ].tile([1, batch_size])

        if self.cfg.update_scheduling_matrix == 'clean':
            to_time_step = torch.zeros_like(to_time_step)

        pbar = tqdm(total=self.update_sampling_timesteps, initial=0,
                    desc="Flow samples updating (stacked frame %d to %d)" % (curr_frame, end_frame + 1))
        if self.update_sampling_timesteps > 1:
            inter_time_steps = int_linspace_tensors(from_time_step, to_time_step, self.update_sampling_timesteps)
            for i in range(self.update_sampling_timesteps):
                xs_pred = self.finetuned_diffusion_model.sample_step(xs_pred, conditions[
                    :end_frame + 1] if conditions is not None else None,
                                                                     inter_time_steps[i], inter_time_steps[i+1],
                                                                     guidance_fn=guidance_fn, )
        else:
            xs_pred = self.finetuned_diffusion_model.sample_step(xs_pred, conditions[:end_frame + 1] if conditions is not None else None,
                                                            from_time_step, to_time_step, guidance_fn=guidance_fn,)
        # if we want clean states, further denoise to zero
        #if self.cfg.update_scheduling_matrix == 'clean':
            #zero_time_step = torch.zeros_like(to_time_step)
            #xs_pred = self.finetuned_diffusion_model.sample_step(xs_pred, conditions[:end_frame + 1] if conditions is not None else None,
                                                                 #to_time_step, zero_time_step, guidance_fn=guidance_fn,)
        pbar.update(1)
        return xs_pred[update_window].clone()

def int_linspace_tensors(x: torch.Tensor, y: torch.Tensor, h: int, rounding="round"):
    """
    Create (h-1) integer tensors z1..z_{h-1} such that
    [x, z1, ..., z_{h-1}, y] approximates a uniform decrease.

    h = number of "hops" from x to y. Total points = h+1.
    rounding: "round" | "floor" | "ceil"
    """
    assert h >= 1
    x = x.to(torch.float32)
    y = y.to(torch.float32)

    t = torch.linspace(0, 1, steps=h + 1, device=x.device, dtype=x.dtype)  # (h+1,)
    pts = x.unsqueeze(0) + (y - x).unsqueeze(0) * t.view(-1, *([1] * x.ndim))  # (h+1, ...)

    if rounding == "round":
        pts = pts.round()
    elif rounding == "floor":
        pts = pts.floor()
    elif rounding == "ceil":
        pts = pts.ceil()
    else:
        raise ValueError("rounding must be 'round', 'floor', or 'ceil'")

    pts = pts.to(torch.int64)

    # Force exact endpoints
    pts[0] = x.to(torch.int64)
    pts[-1] = y.to(torch.int64)

    # Return interior tensors
    return [x.to(torch.int64)] + [pts[k] for k in range(1, h)] + [y.to(torch.int64)]