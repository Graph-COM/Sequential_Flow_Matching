from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from typing import Optional, Any
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from einops import rearrange, repeat, reduce
from ..common.abstract_adaptation_task import AbstractAdaptionTask
from .maze_task import MazeTask
from utils.logging_utils import make_trajectory_images
from time import time


class MazeAdaptionTask(AbstractAdaptionTask, MazeTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)  # suppose to call a Model.init() in the multiple inheritance chain
    
    def _preprocess_batch_fine_tune(self, batch):
        """
        Override to add frame_stack processing for maze2d data.
        The fine-tuning data is in unstacked form, but the model expects frame-stacked input.
        Note that we assume the context is of 1 frame
        """
        # Get batch components
        x_prev, x_next, x_gt, goals, pred_time, pad_mask = batch
        pad_mask = pad_mask[0]
        batch_size, context_pred_length = x_prev.shape[:2]

        x_prev = rearrange(x_prev, "b t c -> t b c").contiguous()
        x_next = rearrange(x_next, "b t c -> t b c").contiguous()
        x_gt = rearrange(x_gt, "b t c -> t b c").contiguous()
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
                x_gt_b = x_gt[pred_time[b].item():, b]
                output_tensor[1:1+x_gt_b.size(0), b] = x_gt_b # replace the new prediction by the real trajectory
            # Vectorized version: create a mask for all time steps that should be replaced
            #time_indices = torch.arange(context_pred_length, device=pred_time.device).unsqueeze(0).expand(batch_size, -1)
            #pred_time_stacked = pred_time // self.frame_stack  # Adjust pred_time for frame-stacked data
            #replacement_mask = time_indices >= pred_time
            #replacement_mask = replacement_mask.T
            #replacement_mask = replacement_mask.view([context_pred_length, batch_size] + [1] * (x_gt.ndim - 2))
            #output_tensor = torch.where(replacement_mask, x_gt[:context_pred_length], output_tensor)
        
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
            x_context = self.pad_init(input_tensor[0]) # [B, ...] to [fs, B, ...]
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
        noise_levels, flow_time = self.model._get_renoise_levels_for_batched_input(input_tensor, torch.ones_like(pred_time),
                                                                                   pad_mask)

        starts = x_gt[pred_time[:, 0]-1, torch.arange(0, batch_size)]
        return input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, pad_mask, goals, starts


    def _construct_guidance(self, start, goal, mask):
        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            mask_ = mask[..., None].tile([1, 1, self.frame_stack])
            mask_ = rearrange(mask_, "t b fs -> (t fs) b", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c

                # guidance weight for observation and action
                # mathematically, one may also try multiplying weight by sqrt(alpha_cum)
                # this means you put higher weight to less noisy terms
                # which might be better but we haven't tried yet
                weight = np.array(
                    [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                    + [1] * h_padded  # try to reach the goal at any horizon
                )
                weight = torch.from_numpy(weight).float().to(self.device)

                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=self.observation_dim // 2).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = torch.cat([dist_o, dist_a], -1)
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack:,1:] = 8
                weight[: self.frame_stack,1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]
                weight = weight * mask_[..., None] # zero out padding region
                #episode_return = -(dist * weight).mean() * 1000
                episode_return = -(dist * weight).mean(dim=[0, -1]).sum() * 1000 / 128
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")
                rewards = pred[:, :, -1]
                weight = np.array([10] * self.frame_stack + [0.997 ** j for j in range(horizon)] + [0] * h_padded)
                weight = torch.from_numpy(weight).float().to(self.device)
                episode_return = rewards * weight[:, None]

            return self.guidance_scale * episode_return
        return goal_guidance

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Override the training step for maze adaptation task.
        """
        # Preprocess batch to get input_tensor, output_tensor, noise_levels, mask_tensor
        input_tensor, output_tensor, noise_levels, flow_time, mask_tensor, pad_mask, goals, starts\
                                                    = self._preprocess_batch_fine_tune(batch)

        #guidance_fn = self._construct_guidance(starts, goals, mask_tensor)
        
        if self.fine_tuning_method == 'flow_matching':
            goals_expanded = goals.unsqueeze(0).repeat(input_tensor.size(0), *([1] * goals.dim()))
            goals_expanded = goals_expanded * pad_mask[..., None] # mask padding entries
            goals_expanded = goals_expanded if self.cfg.finetune_external_cond_dim > 0 else None
            xs_pred, loss, masks = self.model._fine_tuning_flow_forward(input_tensor, output_tensor, noise_levels, flow_time,
                                                           mask_tensor, external_cond=goals_expanded)
        # unstack the masks
        masks_unstacked = repeat(mask_tensor, "t b -> (t fs) b", fs=self.frame_stack).contiguous()
        if self.fine_tuning_method == 'flow_matching':
            loss = self.reweight_loss(loss, masks_unstacked)
        else:
            loss = self.reweight_loss(loss, masks)

        # Log the loss using the same pattern as the original code
        print('batch training loss: %.3f ' % loss)
        if batch_idx % 10 == 0:
            self.log("training/loss", loss)

        # Visualization, including masked out entries
        if self.global_step % 2000 == 0:
            xs_pred_unstacked = self._unstack_and_unnormalize(xs_pred)
            xs_unstacked = self._unstack_and_unnormalize(output_tensor)
            starts_unnorm = self._unnormalize_x(starts).detach().cpu().numpy()
            goals_unnorm = self._unnormalize_x(goals).detach().cpu().numpy()
            #trajectory = xs_pred_unstacked.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8
            images = []
            images_goal = []
            batch_size = input_tensor.size(1)
            plan_horizon = []
            for b in range(min(8, batch_size)):
                mask_last_ind = masks_unstacked[:, b].nonzero(as_tuple=False).max().item() + 1
                plan_horizon.append(mask_last_ind)
                pred_trajectory = xs_pred_unstacked.detach().cpu().numpy()[:mask_last_ind, b:b+1]
                target_trajectory = xs_unstacked.detach().cpu().numpy()[:mask_last_ind, b:b+1]
                images += make_trajectory_images(self.env_id, pred_trajectory, pred_trajectory.shape[1],
                                                 starts_unnorm[b:b+1], goals_unnorm[b:b+1], True)
                images_goal += make_trajectory_images(self.env_id, target_trajectory, target_trajectory.shape[1],
                                                      starts_unnorm[b:b+1], goals_unnorm[b:b+1], True)
            for i, (img, img_g, horizon) in enumerate(zip(images, images_goal, plan_horizon)):
                #self.log_image(f"training_visualization_adaptation/sample_{i}_horizon_{horizon}", Image.fromarray(img))
                #self.log_image(f"training_visualization_adaptation/target_{i}_horizon_{horizon}", Image.fromarray(img_g))
                self.log_image(f"training_visualization_adaptation/sample_{i}", Image.fromarray(img))
                self.log_image(f"training_visualization_adaptation/target_{i}", Image.fromarray(img_g))

        # For fine-tuning format, use the predictions directly
        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,  # This is the updated_prediction
            "xs": output_tensor,  # Use ground truth for visualization
        }

        return output_dict

    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        # conditions are not used for maze planning
        #self.eval_planning_sliding_window(batch, conditions=None, namespace="validation")

        #if not self.save_inference:
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if not self.save_inference:
            # test time: random starting point and a fixed goal
            #self.interact(batch_size, conditions, namespace)
            # manually defined goals for test overfitting
            #obs = batch[0]
            #goal = obs[:, -1]
            #goal = torch.tensor([[1.04, 1.89], [1.04, 5.5]])
            #self.interact(batch_size, conditions, namespace, goal=goal)
            self.interact(batch_size, conditions, namespace)

            # only for overfitting test
            #samples = min(24, batch_size)
            #inference_data = torch.load('./data/maze2d/fine-tuning/df_planning_finetuner_shrinking_horizon.pt')
            #trajectory = inference_data['ground_truth'].to(xs.device)
            #trajectory = self._unnormalize_x(trajectory).cpu()
            #goal = self._unnormalize_x(inference_data['goals'].to(xs.device)).cpu().numpy().tolist()
            #start = trajectory[0].cpu().numpy().tolist()
            #images = make_trajectory_images(self.env_id, trajectory, samples, start, goal, self.plot_end_points)

            #for i, img in enumerate(images):
                #self.log_image(
                    #f"{namespace}_interaction/ground_truth_{i}",
                    #Image.fromarray(img),
                #)

        else:
            # for generating finetuning data: random starting point and goal from dataset
            # manually defined goals for test overfitting
            obs = batch[0]
            goal = obs[:, -1]
            start = obs[:, 0]
            #goal = torch.tensor([[1.04, 1.89], [1.04, 5.5]])
            #goal=None
            num_trials = self.cfg.get('num_gen_trials')
            if num_trials is not None and num_trials > 1:
                batch_size *= self.cfg.num_gen_trials
                goal = goal.repeat_interleave(num_trials, dim=0)
                start = start.repeat_interleave(num_trials, dim=0)
            self.interact(batch_size, conditions, namespace, start=start, goal=goal)


    def eval_planning_sliding_window(self, batch, conditions=None, namespace="validation"):
        """
        Sliding window evaluation for maze planning.
        For each prediction time step, reads observations, actions, and rewards before that step from dataset as context,
        performs planning, and stores the results.
        No environment interaction is needed.
        """
        # batch is the raw batch from dataset: (observations, actions, rewards, nonterminals)
        observations, actions, rewards, nonterminals = batch
        # observations shape: (batch_size, n_frames, observation_dim)
        # actions shape: (batch_size, n_frames, action_dim)
        # rewards shape: (batch_size, n_frames)
        batch_size, n_frames, _ = observations.shape
        observations = observations[..., :self.observation_dim]
        actions = actions[..., :self.action_dim]

        # Normalize observations
        n_context_frames = self.context_frames // self.frame_stack
        obs_mean = self.model.data_mean[:self.observation_dim]
        obs_std = self.model.data_std[:self.observation_dim]
        
        # Set up sliding window parameters
        open_loop_horizon = self.open_loop_horizon
        num_plans = self.episode_len // open_loop_horizon
        
        # Initialize save_inference_buffer if needed
        if self.save_inference:
            # Store ground truth trajectory, start, and goal
            trajectory_bundles = []
            for t in range(n_frames):
                obs_t = observations[:, t, :]  # (batch_size, observation_dim)
                action_t = actions[:, t, :] if t < n_frames - 1 else torch.zeros(batch_size, self.action_dim).to(observations)
                reward_t = rewards[:, t] if t < n_frames - 1 else torch.zeros(batch_size).to(observations)
                bundle = self.make_bundle(obs_t, action_t, reward_t[..., None])
                trajectory_bundles.append(bundle)
            trajectory = torch.stack(trajectory_bundles)  # (n_frames, batch_size, bundle_dim)
            
            self.save_inference_buffer.append([
                trajectory.detach().cpu()   # we do not need conditions for maze planning
            ])
        
        # Sliding window loop: for each prediction time step
        xs_pred_best = None  # Store previous prediction for iterative updates
        xs_context = None
        xs_context_stacked = None
        xs_pred_best_at_ts = []
        xs_gt_at_ts = []
        plan_at_ts = []
        full_plan_at_0s = None

        for step in range(num_plans):
            t_real = open_loop_horizon * step
            
            # Get current observation (start) and goal observation
            # From the diffusion-forcing implementation, planning takes only current observation and goal observation
            current_obs = observations[:, t_real, :]  # (batch_size, observation_dim) - current observation
            goal_obs = observations[:, -1, :]  # (batch_size, observation_dim) - final goal

            # Normalize observations for planning
            current_obs_normalized = ((current_obs - obs_mean[None]) / obs_std[None]).to(self.device)
            goal_obs_normalized = ((goal_obs - obs_mean[None]) / obs_std[None]).to(self.device)
            
            # Prepare xs_best_pred for prediction_update if not the first step
            xs_best_pred_for_update = None
            if step > 0 and xs_pred_best is not None:
                # Convert previous plan to frame-stacked format for prediction_update
                # xs_pred_best is in unstacked format [(t fs), b, c], need to convert to [t, b, (fs c)]
                prev_plan_stacked = rearrange(xs_pred_best, "(t fs) b c -> t b (fs c)", fs=self.frame_stack).contiguous()
                xs_best_pred_for_update = prev_plan_stacked

            if t_real > 0:
                xs_context = observations[:, :t_real, :].clone() # (batch_size, t_real, observation_dim) - context
                xs_context = rearrange(xs_context, "b t c -> t b c").contiguous()
                xs_context_normalized = ((xs_context - obs_mean[None,None]) / obs_std[None,None]).to(self.device)
                xs_context_stacked = rearrange(xs_context_normalized, "(t fs) b c -> t b (fs c)", fs=self.frame_stack).contiguous()
            
            # Use prediction_update for steps > 0, model_sampling for step 0
            use_prediction_update = ((step > 0) and (xs_best_pred_for_update is not None)) and (not self.save_inference)
            plan_normalized = self.plan(
                current_obs_normalized,
                goal_obs_normalized,
                self.episode_len - t_real,
                step,
                conditions,
                use_prediction_update=use_prediction_update,
                xs_best_pred=xs_best_pred_for_update,
                xs_context=xs_context_stacked
            )  # (t, batch_size, bundle_dim)
            plan = self._unnormalize_x(plan_normalized)  # (t, batch_size, bundle_dim)
            
            # Update xs_pred_best
            if xs_pred_best is not None:
                obs_context = observations[:, :t_real, :].clone()
                obs_context_normalized = ((obs_context - obs_mean[None,None]) / obs_std[None,None]).to(self.device)
                obs_context_normalized = rearrange(obs_context_normalized, "b t c -> t b c").contiguous()
                xs_pred_best[:t_real] = obs_context_normalized.clone()
                xs_pred_best[t_real:] = plan_normalized.clone()
            else:
                xs_pred_best = plan_normalized.clone()
            
            # Save to inference buffer if needed
            if self.save_inference:
                self.save_inference_buffer[-1].append([
                    plan.detach().cpu(),
                    torch.tensor([t_real]).tile(batch_size).cpu()
                ])

            plan_at_ts.append(plan.clone())
            xs_pred_best_at_ts.append(plan[:open_loop_horizon].clone())
            xs_gt_at_ts.append(rearrange(observations[:, t_real:t_real + open_loop_horizon, :], "b t c -> t b c").clone())

            if step == 0:
                full_plan_at_0s = plan.clone()

        xs_pred_best_at_ts = torch.stack(xs_pred_best_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)

        # (1) Global MSE: average over all sliding windows, prediction horizons, batch, obs_dim
        mse = (xs_pred_best_at_ts - xs_gt_at_ts).square()
        global_mse = mse.mean().item()
        mse_per_horizon = mse.mean(dim=(0, 2, 3))
        mse_per_window = mse.mean(dim=(1, 2, 3))
        global_rmse = np.sqrt(global_mse)
        rmse_per_horizon = mse_per_horizon.sqrt()
        rmse_per_window = mse_per_window.sqrt()

        self.log(f"{namespace}/mse_sliding_window", global_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log_line_plot(f"{namespace}/mse_per_horizon", mse_per_horizon, "prediction_horizon", "mse", "mse_per_horizon")
        self.log_line_plot(f"{namespace}/mse_per_window", mse_per_window, "window_index", "mse", "mse_per_window")

        start = observations[:, 0, :].detach().cpu()
        goal = observations[:, -1, :].detach().cpu()
        o, _, _ = self.split_bundle(xs_pred_best_at_ts.view(self.episode_len, batch_size, -1))
        o = o.detach().cpu().numpy()[:-1, :16]  # last observation is dummy
        images = make_trajectory_images(self.env_id, o, o.shape[1], start, goal, self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(f"{namespace}_plan/sample_{i}", Image.fromarray(img))

        o, _, _ = self.split_bundle(xs_gt_at_ts.view(self.episode_len, batch_size, -1))
        o = o.detach().cpu().numpy()[:-1, :16]  # last observation is dummy
        images = make_trajectory_images(self.env_id, o, o.shape[1], start, goal, self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(f"{namespace}_gt/sample_{i}", Image.fromarray(img))

        o_pred, _, _ = self.split_bundle(xs_pred_best_at_ts.view(self.episode_len, batch_size, -1))
        o_gt, _, _ = self.split_bundle(xs_gt_at_ts.view(self.episode_len, batch_size, -1))
        o_pred = o_pred.detach().cpu().numpy()[:-1, :16]
        o_gt = o_gt.detach().cpu().numpy()[:-1, :16]
        plan = plan_at_ts[8].detach().cpu().numpy()[:, :16, :]
        o_traj = np.concatenate([o_gt[:400, ...], plan], axis=0)
        images = make_trajectory_images(self.env_id, o_traj, o_traj.shape[1], start, goal, self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(f"{namespace}_pred/sample_{i}", Image.fromarray(img))

        o_full_plan, _, _ = self.split_bundle(full_plan_at_0s.view(self.episode_len, batch_size, -1))
        o_full_plan = o_full_plan.detach().cpu().numpy()[:-1, :16]
        images = make_trajectory_images(self.env_id, o_full_plan, o_full_plan.shape[1], start, goal, self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(f"{namespace}_full_plan_at_0s/sample_{i}", Image.fromarray(img))

    def interact(self, batch_size: int, conditions=None, namespace="validation", start=None, goal=None):
        try:
            import d4rl  # Required to register maze2d environments
            import gym
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("d4rl import not successful, skipping environment interaction. Check d4rl installation.")
            return

        print("Interacting with environment... This may take a couple minutes.")

        def make_env(env_id, goal=None):
            def _init():
                env = gym.make(env_id)
                if goal is not None:
                    env.goal_locations = np.array(goal, dtype=np.float32)
                return env
            return _init

        use_diffused_action = False
        if self.action_dim != 2:
            # https://arxiv.org/abs/2205.09991
            print("Detected reduced observation/action space, using Diffuser like controller.")
        else:
            print("Detected full observation/action space, using MPC controller w/ diffused actions.")
            use_diffused_action = True

        if goal is None:
            envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
        else:
            # manually set goal
            envs = DummyVecEnv([make_env(self.env_id, goal[i:i+1, :self.observation_dim].cpu().numpy()) for i in range(batch_size)])
        envs.seed(0)

        terminate = False
        obs_mean = self.model.data_mean[: self.observation_dim]
        obs_std = self.model.data_std[: self.observation_dim]
        obs = envs.reset()

        if start is not None:
            reset_obs_list = []
            # Set start and goal positions
            for i, env in enumerate(envs.envs):
                # Unwrap to get to the base Mujoco environment
                maze_env = env.unwrapped
                
                # Set start position
                qpos = maze_env.sim.data.qpos.ravel().copy()
                qvel = maze_env.sim.data.qvel.ravel().copy()
                start_pos = start[i, :self.observation_dim].cpu().numpy()
                qpos[:self.observation_dim] = start_pos
                qvel[:] = 0
                maze_env.set_state(qpos, qvel)

                reset_obs = np.zeros(4)
                reset_obs[:self.observation_dim] = start[i, :self.observation_dim].cpu().numpy()
                reset_obs_list.append(reset_obs)
            obs = np.stack(reset_obs_list)

        obs = torch.from_numpy(obs).float().to(self.device)
        start = obs.detach()
        obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

        goal = np.concatenate(envs.get_attr("goal_locations"))
        goal = torch.Tensor(goal).float().to(self.device)
        goal = torch.cat([goal, torch.zeros_like(goal)], -1)
        goal = goal[:, : self.observation_dim]
        goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()

        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []  # actual trajectory
        xs_pred_best = None
        if self.save_inference:
            self.save_inference_buffer.append([goal_normalized.detach().cpu()])

        # run mpc with diffused actions
        while not terminate and steps < self.episode_len:
        #while steps < self.episode_len:
            # planning
            print('\rPlanning for steps %d-%d'
                  % (steps, self.episode_len), end='',
                  flush=True)
            
            xs_best_pred_for_update = None
            # xs_context = None
            # xs_context_stacked = None
            if steps > 0 and xs_pred_best is not None:
                # Convert previous plan to frame-stacked format for prediction_update
                # xs_pred_best is in unstacked format [(t fs), b, c], need to convert to [t, b, (fs c)]
                #prev_plan_stacked = rearrange(xs_pred_best, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)
                # only take the part to be updated
                prev_plan_stacked = rearrange(xs_pred_best, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)
                # Only use the relevant part for the current prediction window
                # -1 to include the last (observed) step. This stacked frame is not going to be used. Just for later size consistency
                xs_best_pred_for_update = prev_plan_stacked[steps//self.frame_stack - 1:]

                # xs_context = torch.stack(trajectory).to(self.device)
                # xs_context = (xs_context - obs_mean[None,None]) / obs_std[None,None]
                # xs_context_stacked = rearrange(xs_context, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)


            # Get plan
            use_prediction_update = ((steps > 0) and (xs_best_pred_for_update is not None)) and (not self.save_inference)

            plan_normalized = self.plan(
                obs_normalized,
                goal_normalized,
                self.episode_len - steps,
                steps//self.open_loop_horizon,
                conditions,
                use_prediction_update=use_prediction_update,
                xs_best_pred=xs_best_pred_for_update,
                xs_context=None
            )
            plan = self._unnormalize_x(plan_normalized)  # (t b c)

            # TODO: for sanity check
            #inference_data = torch.load('./data/maze2d/fine-tuning/df_planning_finetuner_shrinking_horizon.pt')
            #gt = inference_data['ground_truth'].to(plan.device)
            #gt = self._unnormalize_x(gt).to('cpu')
            #plan_gt = inference_data['model_pred'][steps//50].to(plan.device)
            #plan_gt = self._unnormalize_x(plan_gt)
            #if steps == 0:
                # this is a sanity check for fixing the randomness of the first plan
            #print('plan error at step %d is %.6f' % (steps, (plan - plan_gt).square().mean()))
            #plan = plan_gt.clone()
            #plan_normalized = self._normalize_x(plan_gt)
            #gt_trajectory = inference_data['ground_truth'].to(plan.device)
            #gt_trajectory = self._unnormalize_x(gt_trajectory)
            #plan = gt_trajectory[steps+1: steps+self.open_loop_horizon+1]

        # Update xs_pred_best
            if xs_pred_best is not None:
                xs_pred_best[steps:] = plan_normalized.clone()
            else:
                xs_pred_best = plan_normalized.clone()

            # save the current plan into buffer
            if self.save_inference:
                self.save_inference_buffer[-1].append([plan_normalized.detach().cpu(), torch.tensor([steps+1]).tile(batch_size).detach().cpu()])

            # take actions
            # example open_loop_horizon: 50, episode_len: 300, frame_stack: 10
            for t in range(self.open_loop_horizon):
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    plan_vel = plan[t, :, :2] - plan[t - 1, :, :2] if t > 0 else plan[t, :, :2] - obs[:, :2]
                    action = 12.5 * (plan[t, :, :2] - obs[:, :2]) + 1.2 * (plan_vel - obs[:, 2:])
                action = torch.clip(action, -1, 1).detach().cpu()
                obs, reward, done, _ = envs.step(np.nan_to_num(action.numpy()))
                # this calculates the same reward in gym
                reward = np.array([1.0 if np.linalg.norm(obs[i, 0:2] - goal.cpu().detach().numpy()[i]) <= 0.5 else 0.0 for i in range(batch_size)])

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                if done.any():
                    terminate = True
                    break

                obs, reward, done = [torch.from_numpy(item).float() for item in [obs, reward, done]]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle[:, : self.observation_dim])
                # # if reached, we want to stop the trajectory at the reached state
                # if len(trajectory) > 0 and self.save_inference:
                #     # dist_to_goal = np.linalg.norm(obs[:, 0:2] - goal.cpu().detach().numpy()[:, 0:2], axis=1)
                #     # If target is reached in the last step, we want to stop the trajectory at the reached state
                #     reached_mask = torch.from_numpy(reached).unsqueeze(-1)
                #     bundle_obs = bundle[:, : self.observation_dim]
                #     bundle_to_save = torch.where(reached_mask, goal.detach().cpu(), bundle_obs)
                #     trajectory.append(bundle_to_save)
                #     # # Update the reached_strict mask for the next step
                #     # reached_strict = np.logical_or(reached_strict, dist_to_goal <= 0.01)
                # else:
                #     trajectory.append(bundle[:, : self.observation_dim])
                obs = obs.to(self.device)
                obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

                steps += 1

        ## save the real trajectories as the "ground truth"
        self.log(f"{namespace}/clock_time", self.wall_clock * 1000 / self.plan_count)
        trajectory = torch.stack(trajectory)
        # in case we terminate early, we pad the last state
        pad_length = self.n_frames - self.frame_stack - trajectory.shape[0]
        last_state = trajectory[-1:].repeat(pad_length, 1, 1)
        trajectory = torch.cat([trajectory, last_state], dim=0)
        trajectory_with_start = torch.cat([start[None, :, :self.observation_dim].cpu(), trajectory], dim=0)
        trajectory_with_start_normalized = (trajectory_with_start - obs_mean[None].cpu()) / obs_std[None].cpu()
        if self.save_inference:
            self.save_inference_buffer[-1].append(trajectory_with_start_normalized.detach().cpu())
        self.log_line_plot(f"{namespace}/reward", episode_reward, 'sample', 'reward', 'reward')
        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())

        # Visualization
        samples = min(24, batch_size)
        if len(trajectory) > 0:
            #trajectory = torch.stack(trajectory)
            start = start[:, :2].cpu().numpy().tolist()
            goal = goal[:, :2].cpu().numpy().tolist()
            images = make_trajectory_images(self.env_id, trajectory, samples, start, goal, self.plot_end_points)

            for i, img in enumerate(images):
                self.log_image(
                    f"{namespace}_interaction/sample_{i}",
                    Image.fromarray(img),
                )

    def plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, steps: int, conditions: Optional[Any] = None, 
             use_prediction_update: bool = False, xs_best_pred: Optional[torch.Tensor] = None, xs_context: Optional[torch.Tensor] = None):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan of (t, b, c)
        # use_prediction_update: if True, use prediction_update instead of model_sampling
        # xs_best_pred: previous prediction for prediction_update (required if use_prediction_update=True)
        batch_size = start.shape[0]
        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c

                # guidance weight for observation and action
                weight = np.array(
                    [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                    + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                    + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
                )
                # mathematically, one may also try multiplying weight by sqrt(alpha_cum)
                # this means you put higher weight to less noisy terms
                # which might be better but we haven't tried yet
                weight = torch.from_numpy(weight).float().to(self.device)

                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=self.observation_dim // 2).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = torch.cat([dist_o, dist_a], -1)
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack:, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]
                # TODO: here, we should not use .mean() over batch size. We should use .sum()
                # The original batch size 128 with guidance scale 3 and an extra factor 1000
                # equivalent, we use .sum() with guidance scale 3 and an extra factor 1000/128
                #episode_return = -(dist * weight).mean() * 1000
                episode_return = -(dist * weight).mean(dim=[0, -1]).sum() * 1000 / 128
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")
                rewards = pred[:, :, -1]
                weight = np.array([10] * self.frame_stack + [0.997 ** j for j in range(horizon)] + [0] * h_padded)
                weight = torch.from_numpy(weight).float().to(self.device)
                episode_return = rewards * weight[:, None]

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1

        plan_window = [i+1 for i in range(plan_tokens)] # shrinking horizon update strategy, plan to the end


        if use_prediction_update and xs_best_pred is not None:
            # update from the open_loop_horizon step, shrinking horizon
            # plan_window = list(range((self.episode_len - horizon) // self.frame_stack, self.episode_len // self.frame_stack))
            # only take the current start state as context
            init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
            goals_expanded = goal.unsqueeze(0).repeat(xs_best_pred.size(0), *([1] * goal.dim()))
            # Use prediction_update for iterative refinement
            time1 = time()
            if self.cfg.finetune_external_cond_dim > 0:
                # the finetuned model uses classifier-free guidance
                plan = self.model.prediction_update(xs_context=init_token, xs_best_pred=xs_best_pred,
                                                conditions=goals_expanded, prediction_window=plan_window,
                                                n_frames=self.n_frames, batch_size=batch_size,
                                                guidance_fn=None, steps=steps)
            else:
                # the pretrained model uses classifer guidance
                plan = self.model.prediction_update(xs_context=init_token, xs_best_pred=xs_best_pred,
                                                    conditions=conditions, prediction_window=plan_window,
                                                    n_frames=self.n_frames, batch_size=batch_size,
                                                    guidance_fn=guidance_fn)
            time2 = time()
            self.wall_clock += time2 - time1
            self.plan_count += 1
            plan = rearrange(plan, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
        else:
            init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
            # Use model_sampling for initial generation
            plan, _ = self.model.model_sampling(xs_context=init_token, conditions=conditions,
                                                prediction_window=plan_window,
                                                n_frames=self.n_frames, batch_size=batch_size,
                                                guidance_fn=guidance_fn)
            plan = rearrange(plan, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)

        return plan


    def save_inference_data(self, save_buffer):
        import os
        import os.path as osp
        if not osp.exists(self.data_save_dir + '/fine-tuning'):
            os.makedirs(self.data_save_dir + '/fine-tuning')
        algorithm_name = self.cfg._name + '_' + self.update_strategy
        save_path = self.data_save_dir + '/fine-tuning/%s.pt' % algorithm_name
        unzipped = list(zip(*save_buffer))
        goals = torch.cat(unzipped[0], dim=0)
        x_gt = torch.cat(unzipped[-1], dim=1)
        x_pred = []
        t_real = []
        for group in unzipped[1:-1]:
            # group is like ([x_i, t_i] from batch1, [x_i, t_i] from batch2, ...)
            xs, ts = zip(*group)
            xs_all = torch.cat(xs, dim=1)
            ts_all = torch.cat(ts, dim=0)
            #x_pred.append(xs_all.to(torch.float16))
            x_pred.append(xs_all)
            t_real.append(ts_all)

            #x_gt = x_gt.to(torch.float16)

        data_to_save = {'goals': goals, 'ground_truth': x_gt, 'conditions': None, 'model_pred': x_pred, 'pred_time': t_real,
                        'frame_stack': self.frame_stack, 'open_loop_horizon': self.open_loop_horizon,
                        'context_frames': self.context_frames, 'update_strategy': self.update_strategy,
                        'model_wandb_id': self.ckpt_path}

        torch.save(data_to_save, save_path)

    def load_model_weights_only(self, checkpoint_path):
        """
        override to call load_heterog_model_weights_only, since init_mlp has different dim for pretrained and finetuned
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
            self.model.load_heterog_model_weights_only(new_state_dict)
            return True
        return False