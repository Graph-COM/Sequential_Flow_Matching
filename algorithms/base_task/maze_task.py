from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from ..common.abstract_task import AbstractTask
from omegaconf import OmegaConf
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
)
from PIL import Image


# Now this class is used for only flow matching
# because of the pad_tokens parameter in model_sampling
# TODO: modify this class to support diffusion model
class MazeTask(AbstractTask):
    def __init__(self, cfg: DictConfig):
        self.env_id = cfg.env_id
        self.action_dim = len(cfg.action_mean)
        self.observation_dim = len(cfg.observation_mean)
        self.use_reward = cfg.use_reward
        self.unstacked_dim = self.observation_dim + self.action_dim + int(self.use_reward)
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.n_tokens = self.episode_len // cfg.frame_stack + 1
        self.gamma = cfg.gamma
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = np.array(cfg.observation_mean[: self.observation_dim])
        self.observation_std = np.array(cfg.observation_std[: self.observation_dim])
        self.action_mean = np.array(cfg.action_mean[: self.action_dim])
        self.action_std = np.array(cfg.action_std[: self.action_dim])
        self.open_loop_horizon = cfg.get('open_loop_horizon', 1)

        mean = list(self.observation_mean) + list(self.action_mean)
        std = list(self.observation_std) + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        cfg.data_mean = np.array(mean).tolist()
        cfg.data_std = np.array(std).tolist()

        self.padding_mode = cfg.padding_mode
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim
        self.causal = cfg.causal
        self.n_frames = cfg.get('n_frames') if cfg.get('n_frames') is not None else cfg.get('episode_len') + cfg.frame_stack
        self.n_tokens = self.n_frames // cfg.frame_stack

        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0
        self.validation_step_outputs = []
        self.wall_clock = 0
        self.plan_count = 0
        super().__init__(cfg)


    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]

        if (n_frames - 1) % self.frame_stack != 0:
            raise ValueError("Number of frames - 1 must be divisible by frame stack size")

        nonterminals = torch.cat([torch.ones_like(nonterminals[:, : self.frame_stack]), nonterminals[:, :-1]], dim=1)
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)
        bundles = self._normalize_x(self.make_bundle(observations, actions, rewards))  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = 0  # zero out actions and rewards after normalization
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None

        return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

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

        xs_pred, loss = self.model(xs, conditions, masks=masks)

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            o, a, r = self.split_bundle(xs_pred)
            trajectory = o.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8
            images = make_trajectory_images(self.env_id, trajectory, trajectory.shape[1], None, None, False)
            for i, img in enumerate(images):
                self.log_image(
                    f"training_visualization/sample_{i}",
                    Image.fromarray(img),
                )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def on_validation_epoch_end(self, namespace='') -> None:
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if self.guidance_scale == 0:
            namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        #if self.action_dim != 2:
            # self.eval_planning(
            #     batch_size, conditions, horizon, namespace + str(horizon)
            # )  # can run planning without environment installation
            #self.test_overfit_planning(batch, conditions, horizon, namespace + str(horizon))
        self.interact(batch_size, conditions, namespace)  # interact if environment is installation

    def plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, conditions: Optional[Any] = None):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan of (t, b, c)
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

                #episode_return = -(dist * weight).mean() * 1000
                episode_return = -(dist * weight).mean(dim=[0, -1]).sum() * 1000 / 128
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")
                rewards = pred[:, :, -1]
                weight = np.array([10] * self.frame_stack + [0.997 ** j for j in range(h)] + [0] * h_padded)
                weight = torch.from_numpy(weight).float().to(self.device)
                episode_return = rewards * weight[:, None]

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        # pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1

        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")

        plan_window = [i+1 for i in range(plan_tokens)]
        plan, _ = self.model.model_sampling(xs_context=init_token,
                                            conditions=conditions,
                                            prediction_window=plan_window,
                                            n_frames=self.n_frames,
                                            batch_size=batch_size,
                                            guidance_fn=guidance_fn)

        plan = rearrange(plan , "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
        return plan

    def eval_planning(self, batch_size: int, conditions=None, horizon=None, namespace="validation"):
        start, goal = get_random_start_goal(self.env_id, batch_size)

        start_normalized = torch.from_numpy(start).float().to(self.device)
        start_normalized = torch.cat([start_normalized, torch.zeros_like(start_normalized)], -1)
        start_normalized = start_normalized[:, : self.observation_dim]
        start_normalized = self.split_bundle(self._normalize_x(self.make_bundle(start_normalized)))[0]

        goal_normalized = torch.from_numpy(goal).float().to(self.device)
        goal_normalized = torch.cat([goal_normalized, torch.zeros_like(goal_normalized)], -1)
        goal_normalized = goal_normalized[:, : self.observation_dim]
        goal_normalized = self.split_bundle(self._normalize_x(self.make_bundle(goal_normalized)))[0]

        horizon = self.episode_len if horizon is None else horizon
        plan = self.plan(start_normalized, goal_normalized, horizon, conditions)
        plan = self._unnormalize_x(plan)
        plan = plan[self.frame_stack - 1 :]

        # Visualization
        o, _, _ = self.split_bundle(plan)
        o = o.detach().cpu().numpy()[:-1, :16]  # last observation is dummy
        images = make_trajectory_images(self.env_id, o, o.shape[1], start, goal, self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_plan/sample_{i}",
                Image.fromarray(img),
            )

    def test_overfit_planning(self, batch, conditions=None, horizon=None, namespace="validation"):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]
        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]
        start = observations[:, 0, :]
        goal = observations[:, -1, :]

        start_normalized = torch.cat([start, torch.zeros_like(start)], -1)
        start_normalized = start_normalized[:, : self.observation_dim]
        start_normalized = self.split_bundle(self._normalize_x(self.make_bundle(start_normalized)))[0]

        goal_normalized = torch.cat([goal, torch.zeros_like(goal)], -1)
        goal_normalized = goal_normalized[:, : self.observation_dim]
        goal_normalized = self.split_bundle(self._normalize_x(self.make_bundle(goal_normalized)))[0]

        horizon = self.episode_len if horizon is None else horizon
        plan = self.plan(start_normalized, goal_normalized, horizon, conditions)
        plan = self._unnormalize_x(plan)
        plan = plan[self.frame_stack - 1 :]

        # Visualization
        o, _, _ = self.split_bundle(plan)
        o = o.detach().cpu().numpy()[:-1, :16]  # last observation is dummy
        images = make_trajectory_images(self.env_id, o, o.shape[1], start.cpu(), goal.cpu(), self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_plan/sample_{i}",
                Image.fromarray(img),
            )

    def interact(self, batch_size: int, conditions=None, namespace="validation"):
        try:
            import d4rl
            import gym
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("d4rl import not successful, skipping environment interaction. Check d4rl installation.")
            return

        print("Interacting with environment... This may take a couple minutes.")

        use_diffused_action = False
        if self.action_dim != 2:
            # https://arxiv.org/abs/2205.09991
            print("Detected reduced observation/action space, using Diffuser like controller.")
        else:
            print("Detected full observation/action space, using MPC controller w/ diffused actions.")
            use_diffused_action = True

        envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
        envs.seed(0)

        terminate = False
        obs_mean = self.model.data_mean[: self.observation_dim]
        obs_std = self.model.data_std[: self.observation_dim]
        obs = envs.reset()

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

        # run mpc with diffused actions
        while not terminate and steps < self.episode_len:
            # planning
            print(f'\rPlanning for steps %d-%d'
                  % (steps, self.episode_len), end='',
                  flush=True)
            plan = self.plan(obs_normalized, goal_normalized, self.episode_len - steps, conditions)
            plan = self._unnormalize_x(plan)  # (t b c)

            # take actions
            # example open_loop_horizon: 50, episode_len: 300, frame_stack: 10
            for t in range(self.open_loop_horizon):
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    plan_vel = plan[t, :, :2] - plan[t - 1, :, :2] if t > 0 else plan[t, :, :2] - obs[:, :2]
                    action = 12.5 * (plan[t, :, :2] - obs[:, :2]) + 1.2 * (plan_vel - obs[:, 2:])
                    #action = torch.randn_like(action).to(self.device) # sanity check: random actions
                action = torch.clip(action, -1, 1).detach().cpu()
                obs, reward, done, _ = envs.step(np.nan_to_num(action.numpy()))

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                if done.any():
                    terminate = True
                    break

                obs, reward, done = [torch.from_numpy(item).float() for item in [obs, reward, done]]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle)
                obs = obs.to(self.device)
                obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

                steps += 1

        self.log_line_plot(f"{namespace}/reward", episode_reward, 'sample', 'reward', 'reward')
        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())

        # Visualization
        samples = min(16, batch_size)
        trajectory = torch.stack(trajectory)
        start = start[:, :2].cpu().numpy().tolist()
        goal = goal[:, :2].cpu().numpy().tolist()
        images = make_trajectory_images(self.env_id, trajectory, samples, start, goal, self.plot_end_points)

        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")

        return x

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
            self,
            obs: Optional[torch.Tensor] = None,
            action: Optional[torch.Tensor] = None,
            reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)
