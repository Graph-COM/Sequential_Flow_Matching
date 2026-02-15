
import os
from typing import Optional, Any, List
from omegaconf import DictConfig
import numpy as np
from random import random
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from ..common.abstract_task import AbstractTask
from omegaconf import OmegaConf

try: # TODO: fix this later
    from phi.flow import FluidSimulation, DomainBoundary
    from datasets.smoke.evaluate_solver import build_obstacles_pi_64, solver
except:
    pass

class Control2DTask(AbstractTask):
    def __init__(self, cfg: DictConfig):
        self.observation_dim = cfg.observation_shape
        self.control_dim = cfg.control_shape
        self.episode_len = cfg.episode_len

        self.observation_mean = cfg.observation_mean
        self.observation_std = cfg.observation_std
        self.control_mean = cfg.control_mean
        self.control_std = cfg.control_std
        self.rescaler = torch.tensor(cfg.rescaler).reshape(1, 1, 6, 1, 1)
        self.fast_eval = cfg.fast_eval
        self.random_obstacle = cfg.random_obstacle
        self.save_plots = cfg.save_plots
        self.save_plots_dir = cfg.save_plots_dir

        self.open_loop_horizon = cfg.get('open_loop_horizon', 1)


        # Check below are needed?
        cfg.data_mean =self.observation_mean
        cfg.data_std = self.observation_std
        # mean = list(self.observation_mean)
        # std = list(self.observation_std)
        # cfg.data_mean = np.array(mean).tolist()
        # cfg.data_std = np.array(std).tolist()

        self.rescaler = torch.tensor(cfg.rescaler).reshape(1, 1, 6, 1, 1)
        self.padding_mode = cfg.padding_mode
        self.plot_end_points = cfg.plot_start_goal and cfg.guidance_scale != 0
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        # Set bundle_dim based on cfg.x_shape
        if cfg.x_shape[0] == 6:
            self.bundle_dim = "all"
        elif cfg.x_shape[0] == 2:
            self.bundle_dim = "control"
        else:
            raise ValueError(f"Unsupported x_shape first dim: {cfg.x_shape[0]}")

        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.prediction_horizon = cfg.get('prediction_horizon')
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim
        self.causal = cfg.causal
        self.n_frames = cfg.get('n_frames') if cfg.get('n_frames') is not None else cfg.get('episode_len') + cfg.frame_stack
        self.n_tokens = self.n_frames // cfg.frame_stack
        
        self.validation_step_outputs = []

        super().__init__(cfg)


    def _preprocess_batch(self, batch):
        # input example: [batch_sizes, nframes, types, width, height]
        batch, indices = batch[0], batch[1]
        
        if self.bundle_dim == "control":
            # control_x, control_y
            bundles = batch[:,:, -2:] # [batch_sizes, nframes, 2, width, height]
        elif self.bundle_dim == "all":
            # density velocity_x velocity_y smoke control_x control_y
            bundles = batch  
        batch_size, n_frames = bundles.shape[:2]
        masks = torch.ones(n_frames, batch_size).to(bundles.device)

        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()
        
        if self.cfg.external_cond_dim:
            # smoke
            smoke = batch[:,:, 3].unsqueeze(2) # [batch_sizes, nframes, 1, width, height]
            conditions = rearrange(smoke, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
            conditions = conditions.flatten(2, 3).contiguous()
        else:
            conditions = None

        # output shape: [n_frames, batch_size, types, width, height]
        return bundles, conditions, masks, indices

    def training_step(self, batch, batch_idx):
        xs, conditions, masks, _ = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(2, n_tokens + 1, (batch_size,), device=self.device)
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[:, :n_tokens].bool()
            random_terminal = repeat(random_terminal, "b t -> (t fs) b", fs=self.frame_stack)
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.model(xs, conditions, masks)

        loss = self.reweight_loss(loss, weights)
        # print(f"loss: {loss}")

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = xs[self.frame_stack - 1 :]
        xs_pred = xs_pred[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            trajectory = xs_pred.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8

            # images = make_trajectory_images(self.env_id, trajectory, trajectory.shape[1], None, None, False)
            # for i, img in enumerate(images):
            #     self.log_image(
            #         f"training_visualization/sample_{i}",
            #         Image.fromarray(img),
            #     )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def on_validation_epoch_end(self, namespace="validation") -> None:
        """
        Process validation outputs from smoke evaluation using the outputs from fastfast_eval_smoke.
        """
        all_densities = np.concatenate([output_dict['density'] for output_dict in self.validation_step_outputs], axis=0)
        all_velocities = np.concatenate([output_dict['velocity'] for output_dict in self.validation_step_outputs], axis=0)
        all_smoke_outs = np.concatenate([output_dict['smoke_out'] for output_dict in self.validation_step_outputs], axis=0)
        all_controls = np.concatenate([output_dict['control'] for output_dict in self.validation_step_outputs], axis=0)
        
        if self.save_plots:
            from pathlib import Path
            save_plots_dir = Path(self.save_plots_dir)
            save_plots_dir.mkdir(parents=True, exist_ok=True)
            #[batch_sizes, nframes, width, height]
            np.save(save_plots_dir / "all_densities.npy", all_densities)
            #[batch_sizes, nframes, width, height, 2]
            np.save(save_plots_dir / "all_velocities.npy", all_velocities)
            #[batch_sizes, nframes, 1]
            np.save(save_plots_dir / "all_smoke_outs.npy", all_smoke_outs)
            #[batch_sizes, nframes, width, height, 2]
            np.save(save_plots_dir / "all_controls.npy", all_controls)
            print(f"Plots saved to {save_plots_dir}")
            
        # all_smoke_outs: [batch_size, num_windows, 1]
        final_smoke_outs = all_smoke_outs[:, -1].reshape(-1)
        global_smoke_outs = all_smoke_outs.reshape(-1)       
        per_window_smoke_outs = all_smoke_outs.mean(axis=0).squeeze() 

        # Control objective: 1 - <final_smoke>
        control_obj_final = 1.0 - final_smoke_outs.mean()

        # Control objective global: 1 - <all_smoke>
        control_obj_global = 1.0 - global_smoke_outs.mean()

        # Per window (timestep): 1 - <smoke> at each step (mean over all batches)
        per_window_control_obj = 1.0 - per_window_smoke_outs  # (num_windows,)
        title = "control_obj_per_window"
        self.log_line_plot(f"{namespace}/{title}", per_window_control_obj,
                           "window_index", "control_obj", title)

        # Log metrics using log_dict to avoid duplicate logging
        control_obj_dict = {
            "control_obj_final": control_obj_final,
            "control_obj_global": control_obj_global,
        }

        self.log_dict(
            {f"{namespace}/{k}": v for k, v in control_obj_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()



    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _, indices = self._preprocess_batch(batch)
        
        # ASSIGN TARGET CONDITIONS FOR SMOKE VALUE AS 1
        conditions = torch.ones_like(conditions).detach()
        
        _, batch_size, *_ = xs.shape
        #if self.guidance_scale == 0:
            #namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        if self.fast_eval:
            self.fast_eval_smoke(xs, horizon, conditions, indices, namespace)
        else:
            raise ValueError("eval_smoke not yet supported (Very slow).")
            self.eval_smoke(xs, horizon, conditions, indices, namespace)


    def eval_smoke(self, batch: torch.Tensor, horizon=None, conditions=None, indices=None, namespace="validation"):
        """
        Sliding window evaluation for simulation, similar to df_video.py validation_step.
        Uses ground truth context and re-forecasts periodically instead of autoregressive prediction.
        """
        # batch shape: [n_frames, batch_size, types, width, height]
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

        env = solver_env_batch(batch_size, self.random_obstacle, args_general=None)
        env.init_sim(indices)
        
        # the conditions is the state of the environment
        state = batch.clone() # [65, batch, 4, 64, 64]
        velocity_t, density_t = env.solver_reset(state.detach().cpu().numpy()[:,:,:,:,:], self.rescaler)  # [64,64,2], [64,64]        
        density, velocity, control, smoke_out = [density_t], [velocity_t], [], []

        all_smoke_out = []
        while steps < self.episode_len:
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

            conditions_to_go = conditions[steps:end]
            state = self.plan_smoke(init[None], end-t_real, conditions_to_go)

            # rescale for burgers numerical solver
            unnormalized_control = self.parse_state(state)
            
            # Create a list to store all solver_step outputs
            solver_outputs = []
            for i in range(open_loop_horizon):
                c1_t, c2_t = unnormalized_control
                # It's clearer to align argument names vertically and add an explanatory comment.
                # For the timestep argument (t), 'steps' should be used, as it represents the current absolute timestep in the episode,
                # while 'i' is the open-loop substep within the current prediction horizon window.
                density_t, zero_densitys_t, velocity_t, smoke_out_value_t = env.solver_step(
                    density_t_batch=density_t,
                    velocity_t_batch=velocity_t,
                    c1_t_batch=c1_t[i],
                    c2_t_batch=c2_t[i],
                    t=steps+i,  # Use 'steps' to indicate the absolute timestep.
                    bucket_index=1
                )

                # Store all outputs in the list
                solver_outputs.append((density_t, velocity_t, smoke_out_value_t))
                
                if i == 0:
                    density_t_init = density_t.copy()
                    velocity_t_init = velocity_t.copy()
                    smoke_out_value_t_init = smoke_out_value_t.copy()
                    control_t_init = (c1_t[0], c2_t[0])  # Store control at first timestep
                density.append(density_t)
                velocity.append(velocity_t)
                control.append(np.stack([c1_t, c2_t], axis=-1))  # Resulting shape: (batch, 64, 64, 2)
                smoke_out.append(smoke_out_value_t)                

            init = self.update_init(batch, density_t_init, velocity_t_init, smoke_out_value_t_init, control_t_init, steps+1)
            # conditions = self.update_conditions(conditions, solver_outputs, steps+1)
            smoke_outputs = [s[-1] for s in solver_outputs]
            all_smoke_out.append(smoke_outputs)
            steps += 1
            if steps >= self.episode_len:
                break
        
        control_obj = 1 - np.stack(all_smoke_out, axis=0)  # shape: (num_windows, open_loop_horizon, batch_size, 1)

        # 1. Only the final of the last step (last window, last step)
        final_control_obj = control_obj[-1, -1].mean().item()
        self.log(f"{namespace}/control_obj_final", final_control_obj, on_step=False, on_epoch=True, sync_dist=True)

        # 2. Global control objective (mean over all steps and batches)
        global_control_obj = control_obj.mean().item()
        self.log(f"{namespace}/control_obj_global", global_control_obj, on_step=False, on_epoch=True, sync_dist=True)

        # 3. Per sliding window (mean over open_loop_horizon and batch)
        per_window_control_obj = control_obj.mean(axis=(1,2)).squeeze()  # shape: (num_windows,)
        title = "control_obj_per_window"
        self.log_line_plot(f"{namespace}/{title}",
                           per_window_control_obj,
                           "window_index", "control_obj", title)


    def fast_eval_smoke(self, batch: torch.Tensor, horizon=None, conditions=None, indices=None, namespace="validation"):
        """
        Sliding window evaluation for simulation, similar to df_video.py validation_step.
        Uses ground truth context and re-forecasts periodically instead of autoregressive prediction.
        Only evaluates at the next prediction step (i==0) and then replans.
        """
        # batch shape: [n_frames, batch_size, types, width, height]
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

        env = solver_env_batch(batch_size, self.random_obstacle, args_general=None)
        env.init_sim(indices)
        
        # the conditions is the state of the environment
        state = batch.clone() # [65, batch, 4, 64, 64]
        velocity_t, density_t = env.solver_reset(state.detach().cpu().numpy()[:,:,:,:,:], self.rescaler)  # [64,64,2], [64,64]        
        smoke_out_t = state[0,:,3,0,0].unsqueeze(-1).detach().cpu().numpy()
        density, velocity, control, smoke_out = [density_t], [velocity_t], [], [smoke_out_t]

        all_smoke_out = []
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
            state = self.plan_smoke(init[None], end-t_real, conditions_to_go)

            # rescale for burgers numerical solver
            p = 0.1
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

            # # Check for NaN values
            # nan_mask = np.isnan(smoke_out_value_t)

            # # Check if any NaN values exist
            # if nan_mask.any():
            #     print("There are NaN values in the tensor.")
            #     import pdb; pdb.set_trace()
            # else:
            #     print("There are no NaN values in the tensor.")
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
            # if conditions is not None:
            #     print("conditions is not None")
            #     conditions = self.update_conditions(conditions, solver_outputs, steps+1)
            steps += 1
            if steps >= self.episode_len:
                break
        
        density = np.stack(density, axis=1)
        velocity = np.stack(velocity, axis=1)
        smoke_out = np.stack(smoke_out, axis=1)
        control = np.stack(control, axis=1)
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
    
    def plan_smoke(self, init_state: torch.Tensor, horizon: int, conditions: Optional[Any] = None):
        def goal_guidance(x):
            # maximize the goal_guidance(x) function
            # pred shape [n_frames, batch_size, types, width, height]
            pred = rearrange(x, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
            # h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0
            pred = pred * self.rescaler
            #smoke out at target bucket
            guidance_sucess = pred[-1,:,3].mean((1,2))
            #control magnitude
            guidance_energy = pred[:,:,4:].square().mean((0,2,3,4))
            self.w_energy = 0
            episode_return = (guidance_sucess - self.w_energy * guidance_energy).mean() 

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None
        batch_size = init_state.shape[1]

        # plan_tokens: number of tokens (frames) for the plan
        #plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        #pad_tokens = self.n_tokens - plan_tokens - 1
        n_context = init_state.size(0)
        plan_window = [n_context+i for i in range(horizon)]
        plan, _ = self.model.model_sampling(xs_context=init_state, conditions=conditions, prediction_window=plan_window,
                                         n_frames=self.n_frames, batch_size=batch_size, guidance_fn=guidance_fn)
        
        # state_hist shape: (schedules, batch_size, types, width, height) [50, 16, 2, 64, 64]
        # plan = rearrange(plan, "t b (fs c) ...-> (t fs) b c ...", fs=self.frame_stack)
        return plan


    def parse_state(self, state, rand_action_p=0, x_upper=10, x_lower=-10, y_upper=10, y_lower=-2):
        """
        Parses the state tensor at time t, returning only control.
        Output and rescaling depend on self.bundle_dim:
          - if "all": expects state shape [frame, batch, 6, 64, 64] -> control_x [4], control_y [5]
          - if "control": expects state shape [frame, batch, 2, 64, 64] -> control_x [0], control_y [1]
        Returns:
            control: tuple (c1_t, c2_t), each np.ndarray (shape [64, 64])
        """
        rescaler = self.rescaler if self.bundle_dim == "all" else self.rescaler[..., -2:, :, :]
        state = state * rescaler
        
        if self.bundle_dim == "all":
            c1_t, c2_t = state[:, :, 4].cpu().numpy(), state[:, :, 5].cpu().numpy()
        elif self.bundle_dim == "control":
            c1_t, c2_t = state[:, :, 0].cpu().numpy(), state[:, :, 1].cpu().numpy()
        else:
            raise ValueError(f"Unknown bundle_dim: {self.bundle_dim}")
        #if rand_action_p > 0:
            #mask = np.random.rand(*c1_t.shape) < rand_action_p
            #c1_noise = np.random.uniform(x_lower, x_upper, c1_t.shape)
            #c2_noise = np.random.uniform(y_lower, y_upper, c2_t.shape)
            #c1_t = np.where(mask, c1_noise, c1_t)
            #c2_t = np.where(mask, c2_noise, c2_t)
        if np.random.rand() < rand_action_p:
            c1_t = np.random.uniform(x_lower, x_upper, c1_t.shape)
            c2_t = np.random.uniform(y_lower, y_upper, c2_t.shape)

        control = (c1_t, c2_t)
        return control

    def update_init(self, batch, density_t, velocity_t, smoke_out_value_t, control_t_init, steps):
        """
        Updates the init state with the latest density, velocity, smoke_out, and control values.
        Handles both "all" and "control" bundle types.

        Args:
            batch: torch.Tensor, shape [frame, batch, C, H, W]
            density_t: numpy array or torch.Tensor, shape [batch, H, W]
            velocity_t: numpy array or torch.Tensor, shape [batch, H, W, 2]
            smoke_out_value_t: 0-dim array or [batch, 1]
            steps: int, the step number
            control_t_init: tuple of (c1_t, c2_t), each numpy array with shape [batch, H, W]
        Returns:
            pred (updated in-place)
        """
        device = batch.device

        # Convert to tensors as needed
        density_t = torch.as_tensor(density_t, device=device).float()      # [batch, H, W]
        velocity_t = torch.as_tensor(velocity_t, device=device).float()    # [batch, H, W, 2]
        smoke_out_value_t = torch.as_tensor(smoke_out_value_t, device=device).float()
        c1_t, c2_t = control_t_init
        c1_t = torch.as_tensor(c1_t, device=device).float()  # [batch, H, W]
        c2_t = torch.as_tensor(c2_t, device=device).float()  # [batch, H, W]

        # Get shapes and check batch size
        frame, batch_size, C, H, W = batch.shape

        # velocity_t: [batch, H, W, 2] -> [batch, 2, H, W]
        velocity_t = velocity_t.permute(0, 3, 1, 2)  # [batch, 2, H, W]
        if self.bundle_dim == "all":
            rescaler = self.rescaler

            # Prepare density tensor [batch, H, W] -> [batch, H, W]
            density_norm = density_t / rescaler[0, 0, 0]
            # velocity [batch, 2, H, W] -> rescale per channel
            velocity_norm = velocity_t / rescaler[0, 0, 1:3].view(1, 2, 1, 1)
            # smoke_out_value [batch, 1]
            smoke_map = smoke_out_value_t / rescaler[0, 0, 3]
            smoke_map = smoke_map.view(batch_size, 1, 1).expand(batch_size, H, W)  # [batch, H, W]
            # control: normalize using rescaler indices 4 and 5
            control_x_norm = c1_t / rescaler[0, 0, 4]
            control_y_norm = c2_t / rescaler[0, 0, 5]

            # Also update last frame's density, velocity, smoke, and control channels
            init = batch[steps, ...].detach().clone()
            init[:, 0, ...] = density_norm
            init[:, 1, ...] = velocity_norm[:, 0, ...]
            init[:, 2, ...] = velocity_norm[:, 1, ...]
            init[:, 3, ...] = smoke_map
            init[:, 4, ...] = control_x_norm
            init[:, 5, ...] = control_y_norm

        elif self.bundle_dim == "control":
            rescaler = self.rescaler[..., -2:, :, :]  # Use last 2 channels for control
            # Normalize control values
            control_x_norm = c1_t / rescaler[0, 0, 4]
            control_y_norm = c2_t / rescaler[0, 0, 5]
            
            init = batch[steps, ...].detach().clone()
            init[:, 0, ...] = control_x_norm
            init[:, 1, ...] = control_y_norm

        else:
            raise ValueError(f"Unknown bundle_dim: {self.bundle_dim}")
        return init.detach()

    # def update_conditions(self, conditions, solver_outputs, steps):
    #     """
    #     Updates the conditions tensor in place with the latest normalized density, velocity, and smoke_out values.
    #     Updates conditions for all steps based on the solver_outputs list using batch tensor operations.
        
    #     Args:
    #         conditions: torch.Tensor, shape [frame, batch, C, H, W]
    #         solver_outputs: list of tuples, each tuple contains (density_t, velocity_t, smoke_out_value_t)
    #                        from solver_step. Can contain one or more outputs depending on evaluation mode.
    #         steps: int, the starting step number
    #     """
            
    #     device = conditions.device
    #     frame, batch, C, H, W = conditions.shape
    #     rescaler = self.rescaler
        
    #     # Determine valid range
    #     num_steps = len(solver_outputs)
    #     end_step = min(steps + num_steps, frame)
    #     valid_steps = end_step - steps
    #     if valid_steps <= 0:
    #         return conditions
        
    #     # density_t: [batch, H, W], velocity_t: [batch, H, W, 2], smoke_out_value_t: [batch, 1]
    #     density_list = [torch.as_tensor(density_t, device=device).float() 
    #                     for density_t, _, _ in solver_outputs[:valid_steps]]
    #     velocity_list = [torch.as_tensor(velocity_t, device=device).float() 
    #                      for _, velocity_t, _ in solver_outputs[:valid_steps]]
    #     smoke_list = [torch.as_tensor(smoke_out_value_t, device=device).float() 
    #                   for _, _, smoke_out_value_t in solver_outputs[:valid_steps]]
        
    #     density_stacked = torch.stack(density_list, dim=0)  # [num_steps, batch, H, W]
    #     velocity_stacked = torch.stack(velocity_list, dim=0)  # [num_steps, batch, H, W, 2]
    #     smoke_stacked = torch.stack(smoke_list, dim=0)  # [num_steps, batch, 1]
        
    #     # Density: [num_steps, batch, H, W] / scalar
    #     density_norm = density_stacked / rescaler[0, 0, 0]
        
    #     # Velocity: [num_steps, batch, H, W, 2] / [2]
    #     velocity_scale = rescaler[0, 0, 1:3].view(1, 1, 1, 1, 2)  # [1, 1, 1, 1, 2]
    #     velocity_norm = velocity_stacked / velocity_scale  # [num_steps, batch, H, W, 2]
        
    #     # Smoke: [num_steps, batch, 1] -> [num_steps, batch, H, W]
    #     smoke_norm = smoke_stacked / rescaler[0, 0, 3]  # [num_steps, batch, 1]
    #     smoke_norm = smoke_norm.view(valid_steps, batch, 1, 1).expand(valid_steps, batch, H, W)  # [num_steps, batch, H, W]
    #     velocity_norm = velocity_norm.permute(0, 1, 4, 2, 3)  # [num_steps, batch, 2, H, W]
        
    #     # Update conditions[steps:end_step] in batch
    #     conditions[steps:end_step, :, 0] = density_norm  # density
    #     conditions[steps:end_step, :, 1] = velocity_norm[:, :, 0]  # velocity_x
    #     conditions[steps:end_step, :, 2] = velocity_norm[:, :, 1]  # velocity_y
    #     conditions[steps:end_step, :, 3] = smoke_norm  # smoke
        
    #     return conditions



class solver_env_batch():
    """
    Batch version of solver_env, supporting batch processing of multiple data samples
    """
    def __init__(self, batch_size: int, random_obstacles: bool, args_general = None):
        """
        Initialize batch processing environment
        
        Args:
            batch_size: batch size
            random_obstacles: use random obstacles or not
            args_general: general parameter configuration
        """
        super().__init__()
        self.batch_size = batch_size
        self.args_general = args_general
        self.smoke_out_value_t = None
        self.smoke_outs = None

        if random_obstacles is True:
            # Load random obstacles from /data/smoke/random_obstacle.npy
            print("Loading random obstacles from ./data/smoke/random_obstacle.npy")
            obstacle_path = "./data/smoke/random_obstacle.npy"
            self.random_obstacles = np.load(obstacle_path)
        else:
            print("No random obstacles loaded")
            self.random_obstacles = None
        
        # Create independent smoke_outs for each batch element
        self.smoke_outs = [np.zeros((7,), dtype=float) for _ in range(batch_size)]
        
        # Store all sim instances
        self.sims = [None] * batch_size

    def init_sim(self, indices: List[int]):
        """
        Initialize simulation environment for all samples in the batch
        
        Args:
            indices: list of indices, each element is an index corresponding to a sample in the batch
        
        Returns:
            list: list containing all simulation environment objects
        """
        if len(indices) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} indices, got {len(indices)}")
        
        self.sims = []
        for i, index in enumerate(indices):
            random_obstacle = self.random_obstacles[index] if self.random_obstacles is not None else None
            sim = FluidSimulation([63]*2, DomainBoundary([(True, True), (True, True)]), force_use_masks=True)
            build_obstacles_pi_64(sim, random_obstacle)
            self.sims.append(sim)
        
        return self.sims

    def solver_reset(self, data_batch, rescaler):
        """
        Reset solver and unnormalize for all batch samples
        
        Args:
            data_batch: numpy array, shape [n_frames, batch_size, channels, H, W]
            rescaler: rescaler tensor, shape [1, 1, 6, 1, 1]
        
        Returns:
            init_velocities: numpy array, [batch_size, 64, 64, 2]
            init_densities: numpy array, [batch_size, 64, 64]
        """
        batch_size = data_batch.shape[0]
        rescaler_np = rescaler.squeeze().detach().cpu().numpy()
        
        # Normalize velocity: divide each channel
        init_velocities = data_batch[0, :, 1:3, :, :]  # [batch_size, 2, 64, 64]

        for v_idx in range(2):
            init_velocities[:, v_idx, :, :] = init_velocities[:, v_idx, :, :] * rescaler_np[v_idx+1]
        init_velocities = init_velocities.transpose(0, 2, 3, 1)  # [batch_size, 64, 64, 2]
        init_densities = data_batch[0, :, 0, :, :]  # [batch_size, 64, 64]
        init_densities = init_densities * rescaler_np[0]
        # Initialize smoke_outs for each sample
        self.smoke_outs = [np.zeros((7,), dtype=float) for _ in range(batch_size)]
        return init_velocities, init_densities

    def solver_step(self, density_t_batch, velocity_t_batch, c1_t_batch, c2_t_batch, t, bucket_index=1):
        """
        Execute one solver step for all batch samples
        
        Args:
            density_t_batch: numpy array, [batch_size, 64, 64]
            velocity_t_batch: numpy array, [batch_size, 64, 64, 2]
            c1_t_batch: numpy array, [batch_size, 64, 64]
            c2_t_batch: numpy array, [batch_size, 64, 64]
            t: int, current time step
            bucket_index: int, index of the target bucket
        
        Returns:
            density_t: numpy array, [batch_size, 64, 64]
            zero_densitys_t: numpy array, [batch_size, 64, 64]
            velocity_t: numpy array, [batch_size, 64, 64, 2]
            smoke_out_value_t: numpy array, [batch_size, 1]
        """
        batch_size = density_t_batch.shape[0]
        
        # Store batch processing results
        density_results = []
        zero_densitys_results = []
        velocity_results = []
        smoke_out_value_results = []
        
        # Process each batch sample separately
        for b in range(batch_size):
            # Extract data for a single sample
            density_t = density_t_batch[b]
            velocity_t = velocity_t_batch[b]
            c1_t = c1_t_batch[b]
            c2_t = c2_t_batch[b]
            sim = self.sims[b]
            
            # Expand dimensions to match solver function input format
            velocity_t = np.expand_dims(velocity_t, axis=0)
            c1_t = np.expand_dims(c1_t, axis=0)
            c2_t = np.expand_dims(c2_t, axis=0)
            
            # Call solver function
            if t == 0:
                density_t, zero_densitys_t, velocity_t, smoke_out_value_t, self.smoke_outs[b] = solver(
                    sim, velocity_t, density_t, c1_t, c2_t, t, 
                    self.smoke_outs[b], per_timelength=1, bucket_index=bucket_index
                )
            else:
                density_t, zero_densitys_t, velocity_t, smoke_out_value_t, self.smoke_outs[b] = solver(
                    sim, velocity_t, density_t, c1_t, c2_t, t,
                    self.smoke_outs[b], per_timelength=1, bucket_index=bucket_index
                )
            
            # Extract the last frame
            density_t = density_t[-1]
            zero_densitys_t = zero_densitys_t[-1]
            velocity_t = velocity_t[-1]
            
            density_results.append(density_t)
            zero_densitys_results.append(zero_densitys_t)
            velocity_results.append(velocity_t)
            smoke_out_value_results.append(smoke_out_value_t)
        
        # Stack all batch results
        density_batch = np.stack(density_results, axis=0)
        zero_densitys_batch = np.stack(zero_densitys_results, axis=0)
        velocity_batch = np.stack(velocity_results, axis=0)
        smoke_out_value_batch = np.stack(smoke_out_value_results, axis=0)
        
        return density_batch, zero_densitys_batch, velocity_batch, smoke_out_value_batch