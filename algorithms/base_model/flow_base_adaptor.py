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
from .flow_base import FlowMatchingBase


class FlowMatchingAdaptor(AbstractAdaptor, FlowMatchingBase):
    """
    This class is for defining (a) saving offline dataset from a pretrained flow model; (b) finetuning and (c) inference
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
        # self.flow_model already contains the frozen pretrained model
        
        # Create a copy for fine-tuning
        self.finetuned_flow_model = self._create_model_copy()

        print("Full fine-tuning setup complete:")
        print(f"  - Frozen model: {type(self.flow_model).__name__}")
        print(f"  - Fine-tuned model: {type(self.finetuned_flow_model).__name__}")
        print(f"  - Both models have identical architecture and initial weights")

    def _create_model_copy(self):
        """
        Create a deep copy of the diffusion model for fine-tuning.
        
        Returns:
            A copy of the diffusion model with identical weights
        """
        # Import the same model class
        from algorithms.base_model.models.flow import FlowMatching
        
        # Create a new instance with the same configuration
        model_copy = FlowMatching(
            x_shape=self.flow_model.x_shape,
            external_cond_dim=self.flow_model.external_cond_dim,
            is_causal=self.flow_model.is_causal,
            cfg=self.flow_model.cfg,
        )
        
        # Copy the weights from the original model
        model_copy.load_state_dict(self.flow_model.state_dict())
        
        # Ensure the copy is on the same device
        if hasattr(self.flow_model, 'device'):
            model_copy = model_copy.to(self.flow_model.device)
        
        print(f"Created model copy with {sum(p.numel() for p in model_copy.parameters())} parameters")
        
        return model_copy

    def _fine_tuning_flow_forward(self, input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, external_cond=None):
        # Step 1: compute effective "noise" based on noise level (here noise level mean how much noise we added to prev predictions)
        # effective noise: eps_tilde = eps + (sqrt_alphas_cumprod)/(sqrt_one_minus_alphas_cumprod) * (input-output)
        # normalize time into [0,1]
        alpha_renoise = extract(self.flow_model.alpha_coeff, noise_levels, input_tensor.shape)
        alpha_flow = extract(self.flow_model.alpha_coeff, flow_time, input_tensor.shape)
        sigma_renoise = extract(self.flow_model.sigma_coeff, noise_levels, input_tensor.shape)
        sigma_flow = extract(self.flow_model.sigma_coeff, flow_time, input_tensor.shape)
        residual = alpha_renoise * input_tensor - alpha_flow * output_tensor
        eps = torch.randn_like(output_tensor)
        eps = torch.clamp(eps, -self.clip_noise, self.clip_noise)
        eps = sigma_renoise * eps
        eps = eps + residual
        eps = eps / sigma_flow
        mask_tensor = mask_tensor.view(*mask_tensor.shape, *([1] * (eps.ndim - mask_tensor.ndim)))
        eps = eps * mask_tensor # mask out noises to zero padding part

        # Step 2: sample random flow time
        inter_time = self._generate_interpolation_flow_time(output_tensor, flow_time)

        # Step 3: finetune diffusion model's denoising network to match the effective noise
        xs_pred, loss = self.finetuned_flow_model.forward_with_given_noise(
                        output_tensor, external_cond=external_cond, inter_time=inter_time, noise=eps)

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
        #clipped_noise_levels = torch.where(
            #noise_levels < 0,
            #torch.full_like(noise_levels, self.finetuned_flow_model.stabilization_level - 1, dtype=torch.long),
            #noise_levels,
        #)

        # Treating as stabilization would require us to scale with sqrt of alpha_cum
        #orig_input_tensor = input_tensor.clone().detach()
        #scaled_context = self.finetuned_flow_model.q_sample(
            #input_tensor,
            #clipped_noise_levels,
            #noise=torch.zeros_like(input_tensor),
        #)
        #input_tensor = torch.where(
            #self.finetuned_flow_model.add_shape_channels(noise_levels < 0),
            #scaled_context,
            #orig_input_tensor
        #)

        # Get model prediction based on the fine-tuned diffusion model
        if self.fine_tuning_method == 'predictor_denoising':
            # use E(x_0|x_t) estimation
            model_pred = self.finetuned_flow_model.model_predictions(input_tensor, noise_levels, external_cond=external_cond)
            model_pred = model_pred.pred_x_start  # [seq_len, batch_size, C, H, W]

        elif self.fine_tuning_method == 'predictor_residual':
            # directly use model output
            model_pred = self.finetuned_flow_model.model(input_tensor, noise_levels, None,
                                                         is_causal=self.finetuned_flow_model.is_causal,
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
        batched_noise_levels = torch.zeros(context_pred_length, batch_size, device=input_tensor.device)
        batched_flow_time = torch.zeros(context_pred_length, batch_size, device=input_tensor.device)

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

            #context_renoise = (self.timesteps - 1) / self.timesteps
            #context_flow_time = (self.timesteps - 1) / self.timesteps
            context_renoise = 1.0
            context_flow_time = 1.0
            batched_noise_levels = torch.where(context_mask_t, context_renoise, batched_noise_levels)
            batched_flow_time = torch.where(context_mask_t, context_flow_time, batched_flow_time)

        # Vectorized assignment for renoise levels and flow time
        prediction_horizon = self.cfg.prediction_horizon // self.frame_stack
        renoise_levels, flow_time = self._generate_renoise_level_and_flow_time(
            update_horizon=prediction_horizon, device=input_tensor.device)
        batched_noise_levels = self.vectorized_update(batched_noise_levels, context_lengths, renoise_levels)
        batched_flow_time = self.vectorized_update(batched_flow_time, context_lengths, flow_time)
        #for b in range(batch_size):
            #batched_noise_levels[pred_time[b]:pred_time[b]+prediction_horizon] = renoise_levels
            #batched_flow_time[pred_time[b]:pred_time[b] + prediction_horizon] = flow_time

        # Mask out zero padding
        batched_noise_levels = batched_noise_levels * pad_mask.int()
        batched_flow_time = batched_flow_time * pad_mask.int()

        # flow time cannot be exactly zero for numerical concern
        batched_flow_time[batched_flow_time == 0.] = self.numerical_stabilizer

        return batched_noise_levels, batched_flow_time

    def configure_optimizers(self):
        """
        Configure optimizer for direct fine-tuning.
        Train all parameters of the diffusion model.
        """
        if self.fine_tuning_mode:
            # Train all parameters of the diffusion model
            optimizer = torch.optim.AdamW(
                self.finetuned_flow_model.parameters(),
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
            if key.startswith('flow_model.'):
                # Load into frozen model (self.flow_model)
                new_state_dict[key] = value
                # Load into fine-tuned model (self.finetuned_flow_model)
                finetuned_key = key.replace('flow_model.', 'finetuned_flow_model.')
                new_state_dict[finetuned_key] = value
            else:
                # Keep other keys as they are
                new_state_dict[key] = value
            # Load the state dict
        self.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully for fine-tuning")
        return True



    def _generate_interpolation_flow_time(self, xs: torch.Tensor, flow_time: torch.Tensor,
                                            ) -> torch.Tensor:
        """
        Generate flow matching time for training
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

        inter_time = (inter_time * flow_time) # scale [0, 1] to desired [0, flow_time]
        #inter_time = inter_time * (upper_time - lower_time) + lower_time
        inter_time = torch.clip(inter_time, min=lower_time, max=upper_time)

        return inter_time

    def seq2seq_update(self, xs_context, xs_best_pred, conditions, update_window, n_frames, batch_size, diff_hist=False):
        xs_pred = xs_context.clone()
        #xs_pred_hist = [[] for _ in range(len(prediction_window))] if diff_hist else None
        start_frame = update_window[0]

        noise_levels, _ = self._generate_renoise_level_and_flow_time(len(update_window), xs_pred.device)
        noise_levels = torch.cat([torch.zeros([start_frame]).to(noise_levels.device), noise_levels], dim=0)
        noise_levels = noise_levels.view([-1, 1]).tile([1, batch_size])

        xs_to_update = xs_best_pred[update_window]
        input_tensor = torch.cat([xs_pred, xs_to_update], 0)


        if self.fine_tuning_method == 'predictor_denoising':
            # use E(x_0|x_t) estimation
            model_pred = self.finetuned_flow_model.model_predictions(input_tensor, noise_levels, external_cond=conditions)
            model_pred = model_pred.pred_x_start  # [seq_len, batch_size, C, H, W]

        elif self.fine_tuning_method == 'predictor_residual':
            # directly use model output
            model_pred = self.finetuned_flow_model.model(input_tensor, noise_levels, None,
                                                         is_causal=self.finetuned_flow_model.is_causal,
                                                         external_cond=conditions)
            model_pred = model_pred + input_tensor

        return model_pred[update_window].clone()

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
            # fetch old prediction
            xs_to_update = xs_best_pred[curr_frame:curr_frame + horizon]
            # renoise: the starting point z0 of the adaptor flow
            chunk = self.flow_model.q_sample(xs_to_update, renoise_levels[update_idx:update_idx+horizon, None].tile((1, batch_size)))
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            #start_frame = max(0, curr_frame + horizon - len(pre))

            # scheduling matrix
            scheduling_matrix = self._generate_update_scheduling_matrix(flow_time[update_idx:update_idx+horizon].cpu().numpy())

            pbar = tqdm(total= scheduling_matrix.shape[0]-1, initial=0,
                        desc="Flow samples updating (stacked frame %d to %d)" % (curr_frame, curr_frame+horizon-1))

            for m in range(scheduling_matrix.shape[0] - 1):
                from_time_step = np.concatenate((np.zeros((curr_frame,)), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_time_step = np.concatenate(
                    (
                        np.zeros((curr_frame,)),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_time_step = torch.from_numpy(from_time_step).to(self.device, xs_pred.dtype)
                to_time_step = torch.from_numpy(to_time_step).to(self.device, xs_pred.dtype)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                #xs_pred[start_frame:] = self.flow_model.sample_step(
                #xs_pred[start_frame:],
                #conditions[start_frame: curr_frame + horizon],
                #from_noise_levels[start_frame:],
                #to_noise_levels[start_frame:],
                #)
                xs_pred = self.finetuned_flow_model.sample_step(xs_pred, conditions[:curr_frame+horizon] if conditions is not None else None,
                                                                from_time_step, to_time_step)

                pbar.update(1)
                pbar.set_postfix(m=m)
            curr_frame += horizon
        return xs_pred[update_window].clone()

    def _generate_update_scheduling_matrix(self, flow_time):
        # TODO: to explore this design space of scheduling
        #flow_time = (flow_time * self.cfg.flow.sampling_timesteps).astype(np.int32)
        #flow_time = (flow_time * self.update_sampling_timesteps).astype(np.int32)
        match self.cfg.get('update_scheduling_matrix'):
            #case 'pyramid':
                #return self._generate_update_pyramid_scheduling_matrix(flow_time, self.uncertainty_scale)
            #case 'full_sequence':
                #return self._generate_update_pyramid_scheduling_matrix(flow_time, uncertainty_scale=0)
            #case 'autoregressive':
                #return self._generate_update_autoregressive_scheduling_matrix(flow_time)
            case 'shortest':
                return self._generate_update_shortest_scheduling_matrix(flow_time)

    #def _generate_update_pyramid_scheduling_matrix(self, flow_time, uncertainty_scale: float):
        #horizon = len(flow_time)
        #height = flow_time.max() + int((horizon - 1) * uncertainty_scale) + 1
        #scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        #for m in range(height):
            #for t in range(horizon):
                #scheduling_matrix[m, t] = flow_time[t] + int(t * uncertainty_scale) - m
                #scheduling_matrix[m, t] = np.clip(scheduling_matrix[m, t], 0, flow_time[t])

        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
        #return scheduling_matrix

    #def _generate_update_autoregressive_scheduling_matrix(self, flow_time):
        #horizon = len(flow_time)
        #flow_time_cumsum = np.cumsum(flow_time)
        #height = sum(flow_time) + 1
        #scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        #for m in range(height):
            #for t in range(horizon):
                #scheduling_matrix[m, t] = flow_time_cumsum[t] - m
                #scheduling_matrix[m, t] = np.clip(scheduling_matrix[m, t], 0, flow_time[t])

        #return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
        #return scheduling_matrix

    def _generate_update_shortest_scheduling_matrix(self, flow_time):
        horizon = len(flow_time)
        #height = flow_time.min() + 1
        height = self.update_sampling_timesteps + 1
        scheduling_matrix = np.zeros((height, horizon))
        for t in range(horizon):
            schedule_at_t = np.linspace(flow_time[t], 0, height)
            #schedule_at_t = np.rint(schedule_at_t).astype(np.int32)
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
        renoise_levels = (self.cfg.renoise_level * renoise_factor_of_lead_time) / self.timesteps # normalized to [0, 1]
        flow_time = (self.cfg.flow_time * flow_time_factor_of_lead_time) / self.timesteps # normalized to [0, 1]

        # if predict last frame from noise, then set renoise level and flow time to 1
        if self.cfg.last_frame_update == 'from_noise':
            renoise_levels[-1], flow_time[-1] = 1.0, 1.0

        # numerical stabilizer
        #lower_bound, upper_bound = 1e-4, 1-1e-4
        #renoise_levels = renoise_levels * (upper_bound - lower_bound) + lower_bound
        #flow_time = flow_time * (upper_bound - lower_bound) + lower_bound
        return renoise_levels, flow_time

    def sample_t(self, dim, device):
        if self.cfg.inter_time_dist == 'uniform':
            return torch.rand(dim, device=device)
        elif self.cfg.inter_time_dist == 'log_norm':
            mu, sigma = self.cfg.inter_time_mean, self.cfg.inter_time_std
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

        noise_levels = torch.tensor([i / prediction_horizon for i in range(1, prediction_horizon + 1)]).to(xs_context.device)
        # if this is the very first update step, renoise the previous prediction.
        if (update_window[0] == (self.cfg.context_frames // self.cfg.frame_stack) + 1 and global_step is None) or global_step == 1:
            xs_pred = self.flow_model.q_sample(xs_best_pred[update_window[:-1]], noise_levels[:-1][:, None].tile((1, batch_size)))
        else:
            xs_pred = xs_best_pred[update_window[:-1]].clone()


        # append a Gaussian noise to the end
        xs_pred = torch.cat([xs_pred, torch.randn_like(xs_pred)[0:1]], dim=0)
        # add context
        xs_pred = torch.cat([xs_context, xs_pred], dim=0)

        # from_noise and to_noise

        from_time_step = torch.cat([torch.zeros((curr_frame,)).to(xs_pred.device), noise_levels])[
                :, None
            ].tile([1, batch_size])
        to_time_step = torch.cat([torch.zeros((curr_frame+1,)).to(xs_pred.device), noise_levels[:-1]])[
            :, None
        ].tile([1, batch_size])

        pbar = tqdm(total=1, initial=0,
                    desc="Flow samples updating (stacked frame %d to %d)" % (curr_frame, end_frame+1))
        xs_pred = self.finetuned_flow_model.sample_step(xs_pred, conditions[:end_frame+1],
                                                        from_time_step, to_time_step)
        pbar.update(1)
        return xs_pred[update_window].clone()