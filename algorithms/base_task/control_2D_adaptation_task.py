from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from typing import Optional, Any
import torch
import numpy as np
from tqdm import tqdm
from ..common.abstract_adaptation_task import AbstractAdaptionTask
from .control_2D_tasks import Control2DTask
from einops import rearrange

class Control2DAdaptionTask(AbstractAdaptionTask, Control2DTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)  # suppose to call a Model.init() in the multiple inheritance chain

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Override the training step for maze adaptation task.
        """
        # Preprocess batch to get input_tensor, output_tensor, noise_levels, mask_tensor
        input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, pad_mask \
            = self._preprocess_batch_fine_tune(batch)

        if self.fine_tuning_method == 'flow_matching':
            conditions = torch.ones_like(output_tensor)
            conditions = conditions[:, :, 0:1]
            xs_pred, loss, masks = self.model._fine_tuning_flow_forward(input_tensor, output_tensor, noise_levels,
                                                                        flow_time,
                                                                        mask_tensor, external_cond=conditions)
        # unstack the masks
        # remove loss on smoke out for sanity check
        #smoke_out_mask = torch.ones_like(loss)
        #smoke_out_mask[:, :, 3] = 0.
        #smoke_out_mask = smoke_out_mask.bool()
        #loss = loss * ~smoke_out_mask
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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, _, indices = self._preprocess_batch(batch)
        
        # ASSIGN TARGET CONDITIONS FOR SMOKE VALUE AS 1
        conditions = torch.ones_like(conditions).detach()
        
        _, batch_size, *_ = xs.shape
        
        if self.save_inference:
            # Initialize save buffer for current batch: [ground_truth, conditions]
            if torch.is_tensor(indices):
                sim_ids = indices.detach().cpu().long()
            else:
                sim_ids = torch.as_tensor(indices, dtype=torch.long)
            # no need to save xs. Should save the actual trajectory
            #self.save_inference_buffer.append([xs.detach().cpu(), sim_ids])
            self.save_inference_buffer.append([sim_ids])

        # Call the parent validation step which handles fast_eval_smoke
        # We'll override fast_eval_smoke to add inference saving logic
        horizon = self.episode_len
        if self.fast_eval:
            self.fast_eval_smoke_with_inference(batch=xs, horizon=horizon, conditions=conditions, indices=indices, namespace=namespace)
        else:
            raise ValueError("eval_smoke not yet supported (Very slow).")

    def fast_eval_smoke_with_inference(self, batch: torch.Tensor, horizon=None, conditions=None, indices=None, namespace="validation"):
        """
        Sliding window evaluation for simulation with inference saving capability.
        Based on Control2DTask.fast_eval_smoke but adds inference data saving.
        Only evaluates at the next prediction step (i==0) and then replans.
        """
        # batch shape: [n_frames, batch_size, types, width, height]
        # print(indices)
        init = batch[0, ...]
        n_frames, batch_size, *_ = batch.shape
        self.rescaler = self.rescaler.to(batch.device)
        steps = 0
        # Set up sliding window parameters
        prediction_horizon = self.prediction_horizon * self.frame_stack
        n_context_frames = self.context_frames // self.frame_stack
        prediction_horizon = prediction_horizon // self.frame_stack
        num_sliding_windows = n_frames - prediction_horizon - n_context_frames + 1
        open_loop_horizon = self.open_loop_horizon
        # assert open_loop_horizon <= prediction_horizon
        prediction_horizon = max(prediction_horizon, open_loop_horizon)

        # Import solver_env_batch from parent module
        from .control_2D_tasks import solver_env_batch
        # import pdb; pdb.set_trace()
        env = solver_env_batch(batch_size, self.random_obstacle, args_general=None)
        env.init_sim(indices)
        
        # the conditions is the state of the environment
        state = batch.clone() # [65, batch, 4, 64, 64]
        velocity_t, density_t = env.solver_reset(state.detach().cpu().numpy()[:,:,:,:,:], self.rescaler)  # [64,64,2], [64,64]
        smoke_out_t = state[0,:,3,0,0].unsqueeze(-1).detach().cpu().numpy()
        density, velocity, control, smoke_out = [density_t], [velocity_t], [], [smoke_out_t]

        # initialize the best estimation of xs
        xs_best_pred = torch.zeros_like(batch)

        for steps in tqdm(range(self.episode_len), desc="fast_eval_smoke steps"):
            # add noise to target_step
            """
            x: torch.Tensor - The current noisy sample tensor. This is the input that needs to be denoised.
            external_cond: Optional[torch.Tensor] - Optional external conditioning information (like goal states, constraints, etc.) that guides the generation process.
            curr_noise_level: torch.Tensor - The current noise level (timestep) in the diffusion process. This indicates how much noise is currently present.
            next_noise_level: torch.Tensor - The target noise level for the next step. The method will denoise from curr_noise_level to next_noise_level.
            guidance_fn: Optional[Callable] = None - An optional function that provides additional guidance during sampling (e.g., goal-directed guidance for planning tasks).
            """
            t_real = steps + n_context_frames
            end = min(t_real + prediction_horizon, self.episode_len + n_context_frames)

            if conditions is not None:
                conditions_to_go = conditions[steps:end]
            else:
                conditions_to_go = None
            # Generate plan/prediction
            state = self.plan_smoke(init[None], end-t_real, steps, conditions_to_go,
                                    prev_state=xs_best_pred[steps:end] if steps > 0 else None)
            # import pdb; pdb.set_trace()
            # update best estimation so far
            xs_best_pred[t_real - 1] = init.clone()
            xs_best_pred[t_real:end] = state.clone()
            # Save inference data if enabled
            if self.save_inference:
                # Save the predicted state (plan) and the prediction time
                self.save_inference_buffer[-1].append([
                    state.detach().cpu(),
                    torch.tensor([t_real]).tile(batch_size).detach().cpu()
                ])

            # rescale for burgers numerical solver
            p = 0.1 if not self.save_inference else 0.0 # noisy action
            unnormalized_control = self.parse_state(state, rand_action_p=p)

            # Only evaluate at i==0 (next prediction step), then replan
            c1_t, c2_t = unnormalized_control

            density_t, zero_densitys_t, velocity_t, smoke_out_value_t = env.solver_step(
                density_t_batch=density_t,
                velocity_t_batch=velocity_t,
                c1_t_batch=c1_t[0],
                c2_t_batch=c2_t[0],
                t=steps,  # Use 'steps' to indicate the absolute timestep.
                bucket_index=1
            )

            # Store output for i==0 only
            solver_outputs = [(density_t, velocity_t, smoke_out_value_t)]
            density_t_init = density_t.copy()
            velocity_t_init = velocity_t.copy()
            smoke_out_value_t_init = smoke_out_value_t.copy()
            
            density.append(density_t)
            velocity.append(velocity_t)
            control.append(np.stack([c1_t[0], c2_t[0]], axis=-1))  # Resulting shape: (batch, 64, 64, 2)
            smoke_out.append(smoke_out_value_t)                
            control_t_init = (c1_t[0], c2_t[0])  # Control at first timestep
            init = self.update_init(batch, density_t_init, velocity_t_init, smoke_out_value_t_init, control_t_init, steps+1)

            steps += 1
            if steps >= self.episode_len:
                break
        
        # Store validation outputs for metrics (same as base class)
        density = np.stack(density, axis=1)
        velocity = np.stack(velocity, axis=1)
        smoke_out = np.stack(smoke_out, axis=1)
        control = np.stack(control, axis=1)
        # save the true trajectory
        if self.save_inference:
            density_exp = density[..., None]
            smoke_out_exp = smoke_out[..., None, None, :]
            smoke_out_exp = np.broadcast_to(smoke_out_exp, (smoke_out.shape[0], smoke_out.shape[1], density.shape[2],
                                                        density.shape[3], smoke_out.shape[-1]))
            control_exp = np.concatenate((np.zeros_like(control[:, 0:1]), control), axis=1)
            state_trajectory = np.concatenate((density_exp, velocity, smoke_out_exp, control_exp), axis=-1)
            state_trajectory = torch.tensor(state_trajectory.transpose(1, 0, 4, 2, 3))
            state_trajectory = state_trajectory / self.rescaler.to(state_trajectory.device)
            self.save_inference_buffer[-1].append(torch.tensor(state_trajectory))
        # Store all collected data in a dictionary
        output_dict = {
            'density': density,  # list of arrays
            'velocity': velocity,  # list of arrays
            'smoke_out': smoke_out,  # list of arrays
            'control': control,  # list of arrays
            'batch_size': batch_size,
            'episode_len': self.episode_len,
            'open_loop_horizon': 1  # fast_eval only uses i==0
        }
        self.validation_step_outputs.append(output_dict)

    def plan_smoke(self, init_state: torch.Tensor, horizon: int, steps: int, conditions: Optional[Any] = None,
                   prev_state: torch.Tensor = None):
        def goal_guidance(x):
            # maximize the goal_guidance(x) function
            # pred shape [n_frames, batch_size, types, width, height]
            pred = rearrange(x, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
            # h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0
            pred = pred * self.rescaler
            # smoke out at target bucket
            guidance_sucess = pred[-1, :, 3].mean((1, 2))
            # control magnitude
            guidance_energy = pred[:, :, 4:].square().mean((0, 2, 3, 4))
            self.w_energy = 0
            episode_return = (guidance_sucess - self.w_energy * guidance_energy).mean()

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None
        batch_size = init_state.shape[1]

        # plan_tokens: number of tokens (frames) for the plan
        # plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        # pad_tokens = self.n_tokens - plan_tokens - 1
        n_context = init_state.size(0)
        plan_window = [n_context + i for i in range(horizon)]
        if prev_state is not None:
            last_frame_pad = True if horizon == self.prediction_horizon else False
            plan = self.model.prediction_update(xs_context=init_state, xs_best_pred=prev_state, conditions=conditions,
                                                prediction_window=plan_window, n_frames=self.n_frames,
                                                batch_size=batch_size, guidance_fn=guidance_fn, last_frame_pad=last_frame_pad, steps=steps)
        else:
            plan, _ = self.model.model_sampling(xs_context=init_state, conditions=conditions, prediction_window=plan_window,
                                            n_frames=self.n_frames, batch_size=batch_size, guidance_fn=guidance_fn)

        # state_hist shape: (schedules, batch_size, types, width, height) [50, 16, 2, 64, 64]
        # plan = rearrange(plan, "t b (fs c) ...-> (t fs) b c ...", fs=self.frame_stack)
        return plan

    def save_inference_data(self, save_buffer):
        import os
        import os.path as osp
        # I have to hard-code this save_dir due to access limit
        self.data_save_dir = '/usr/scratch/yhuang903/diffusion_iterative_update/data/smoke'
        if not osp.exists(self.data_save_dir + '/fine-tuning'):
            os.makedirs(self.data_save_dir + '/fine-tuning')
        # TODO: to save space, cast to float16 and use np.savez_compressed
        algorithm_name = self.cfg._name + '_' + self.update_strategy
        unzipped = list(zip(*save_buffer))

        x_gt = torch.cat(unzipped[-1], dim=1)
        sim_ids_list = []
        for sim_id_batch in unzipped[0]:
            if isinstance(sim_id_batch, torch.Tensor):
                sim_ids_list.append(sim_id_batch.detach().cpu().long())
            else:
                sim_ids_list.append(torch.as_tensor(sim_id_batch, dtype=torch.long))
        sim_ids = torch.cat(sim_ids_list, dim=0)
        pred_start_id = 1

        x_pred = []
        t_real = []
        print('update_strategy: ', self.update_strategy)
        print('model_wandb_id: ', self.ckpt_path)
        for group in unzipped[pred_start_id:-1]:
            # group is like ([x_i, t_i] from batch1, [x_i, t_i] from batch2, ...)
            xs, ts = zip(*group)
            xs_all = torch.cat(xs, dim=1)
            ts_all = torch.cat(ts, dim=0)
            x_pred.append(xs_all.to(torch.float16))
            t_real.append(ts_all)

        sim_min = int(sim_ids.min().item())
        sim_max = int(sim_ids.max().item())
        save_filename = f"{algorithm_name}_sim{sim_min:06d}-{sim_max:06d}.pt"
        save_path = osp.join(self.data_save_dir, 'fine-tuning', save_filename)

        x_gt = x_gt.to(torch.float16)


        torch.save({'ground_truth': x_gt,
                    'model_pred': x_pred,
                    'pred_time': t_real,
                    'sim_ids': sim_ids,  # sim_ids for all data (same for ground truth and predictions)
                    'frame_stack': self.frame_stack,
                    'open_loop_horizon': self.open_loop_horizon,
                    'context_frames': self.context_frames,
                    'update_strategy': self.update_strategy,
                    'model_wandb_id': self.ckpt_path},
                   save_path)

    def _preprocess_batch_fine_tune(self, batch):
        """
        Override to add frame_stack processing for maze2d data.
        The fine-tuning data is in unstacked form, but the model expects frame-stacked input.
        Note that we assume the context is of 1 frame
        """
        # Get batch components
        x_prev, x_next, x_gt, conditions, pred_time, pad_mask = batch
        pad_mask = pad_mask[0]
        batch_size, context_pred_length = x_prev.shape[:2]

        x_prev = rearrange(x_prev, "b t ... -> t b ...").contiguous()
        x_next = rearrange(x_next, "b t ... -> t b ...").contiguous()
        x_gt = rearrange(x_gt, "b t ... -> t b ...").contiguous()
        pad_mask = rearrange(pad_mask, "b t -> t b").contiguous()

        # Continue with the rest of the preprocessing from parent class
        # Define input/output tensor
        input_tensor = x_prev
        if self.fine_tuning_teacher_forcing == 'updated_prediction':
            output_tensor = x_next
        elif self.fine_tuning_teacher_forcing == 'ground_truth':
            output_tensor = input_tensor.clone()
            # TODO: find a vectorized implementation of this
            for b in range(batch_size):
                x_gt_b = x_gt[pred_time[b].item():pred_time[b].item() + self.cfg.prediction_horizon, b]
                output_tensor[1:1 + x_gt_b.size(0), b] = x_gt_b  # replace the new prediction by the real trajectory
            # Vectorized version: create a mask for all time steps that should be replaced
            # time_indices = torch.arange(context_pred_length, device=pred_time.device).unsqueeze(0).expand(batch_size, -1)
            # pred_time_stacked = pred_time // self.frame_stack  # Adjust pred_time for frame-stacked data
            # replacement_mask = time_indices >= pred_time
            # replacement_mask = replacement_mask.T
            # replacement_mask = replacement_mask.view([context_pred_length, batch_size] + [1] * (x_gt.ndim - 2))
            # output_tensor = torch.where(replacement_mask, x_gt[:context_pred_length], output_tensor)

        # Mask out zero padding
        pad_mask_ex = pad_mask.view(*pad_mask.shape, *([1] * (input_tensor.ndim - pad_mask.ndim))).float()
        input_tensor = input_tensor * pad_mask_ex
        output_tensor = output_tensor * pad_mask_ex

        # Define update_mask tensor
        mask_tensor = pad_mask.clone()
        if not self.train_on_context:
            # assume only the first frame is the context
            mask_tensor[0, :] = False

        # now we are ready to consider frame stack
        if self.frame_stack > 1:
            # expand context from 1 frame to self.frame_stack frames
            x_context = self.pad_init(input_tensor[0])  # [B, ...] to [fs, B, ...]
            input_tensor = torch.cat([x_context, input_tensor[1:]], dim=0)
            output_tensor = torch.cat([x_context, output_tensor[1:]], dim=0)
            mask_tensor = torch.cat([self.pad_init(mask_tensor[0]), mask_tensor[1:]], dim=0)
            pad_mask = torch.cat([self.pad_init(pad_mask[0]), pad_mask[1:]], dim=0)
            # reshape from [(t fs), B, C] to [t, B, (c fs)]
            input_tensor = rearrange(input_tensor, "(t fs) b c -> t b (fs c)", fs=self.frame_stack).contiguous()
            output_tensor = rearrange(output_tensor, "(t fs) b c -> t b (fs c)", fs=self.frame_stack).contiguous()
            mask_tensor = rearrange(mask_tensor, "(t fs) b -> t b fs", fs=self.frame_stack).contiguous()
            mask_tensor = mask_tensor.any(dim=-1)
            pad_mask = rearrange(pad_mask, "(t fs) b -> t b fs", fs=self.frame_stack).contiguous()
            pad_mask = pad_mask.any(dim=-1)

        # Get re-noise levels
        noise_levels, flow_time = self.model._get_renoise_levels_for_batched_input(input_tensor,
                                                                                   torch.ones_like(pred_time),
                                                                                   pad_mask)

        return input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, pad_mask


