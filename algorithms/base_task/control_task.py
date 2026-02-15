from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from ..common.abstract_task import AbstractTask
from omegaconf import OmegaConf
from datasets.pde.solver import burgers_numeric_solve_free

class ControlTask(AbstractTask):
    def __init__(self, cfg: DictConfig):
        self.observation_dim = cfg.observation_shape[0]
        self.control_dim = cfg.observation_shape[0]
        self.episode_len = cfg.episode_len
        self.unstacked_dim = self.observation_dim + self.control_dim
        cfg.x_shape = (self.unstacked_dim,)
        # self.gamma = cfg.gamma
        # mean = (obs_mean, control_mean)
        self.observation_mean = np.load(cfg.observation_mean)[: self.observation_dim]
        self.observation_std = np.load(cfg.observation_std)[: self.observation_dim]
        self.control_mean = np.load(cfg.control_mean)[: self.control_dim]
        self.control_std = np.load(cfg.control_std)[: self.control_dim]
        mean = list(self.observation_mean) + list(self.control_mean)
        std = list(self.observation_std) + list(self.control_std)
        cfg.data_mean = np.array(mean).tolist()
        cfg.data_std = np.array(std).tolist()

        self.rescaler = cfg.rescaler
        self.open_loop_horizon = cfg.get('open_loop_horizon', 1)
        self.padding_mode = cfg.padding_mode
        self.plot_end_points = cfg.plot_start_goal and cfg.guidance_scale != 0
        self.cfg = cfg
        self.x_shape = cfg.x_shape
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
        # input example: [batch_sizes, type, nframes, observation_dim]
        u = batch[:, 0]         # [batchsizes, nframes, observation_dim]
        controls = batch[:, 1]   # [batchsizes, nframes, observation_dim]

        observations = u[..., : self.observation_dim]
        controls = controls[..., : self.control_dim]

        batch_size, n_frames = u.shape[:2]
        masks = torch.ones(n_frames, batch_size).to(u.device)

        # check this do we need to remove the last control?
        controls = controls[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)

        bundles = self._normalize_x(self.make_bundle(observations, controls))
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))
        init_bundle[:, self.observation_dim :] = 0  # zero out controls after normalization
        init_bundle = self.pad_init(init_bundle, batch_first=True)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()
        if self.cfg.external_cond_dim:
            # u_target is the target state
            conditions =  batch[:, 2] # [batchsizes, nframes, observation_dim]
            # normalize the conditions
            obs_mean = torch.from_numpy(self.observation_mean).to(conditions.device)
            obs_std = torch.from_numpy(self.observation_std).to(conditions.device)
            # conditions: [batch_size, n_frames, observation_dim]
            conditions = (conditions - obs_mean) / obs_std
            conditions = rearrange(conditions, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
            conditions = conditions.flatten(2, 3).contiguous()
        else:
            raise ValueError("Control conditions are required")

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

        xs_pred, loss = self.model(xs, conditions, masks)

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            observation, control = self.split_bundle(xs_pred)
            trajectory = observation.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8
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

    def on_validation_epoch_end(self, namespace='') -> None:
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if self.guidance_scale == 0:
            namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        self.eval_pde(xs, horizon, conditions, namespace)

    def eval_pde(self, batch: torch.Tensor, horizon=None, conditions=None, namespace="validation"):
        trajectory = []
        all_plan_hist = []
        # batch shape: (n_frames, batch_size, observation_dim(u,f))
        # u_init shape: (batch_size, observation_dim(u,f))
        u_init = batch[0, ...]
        # the state in batch[t:, ...] is the same as conditions[t]
        u_target = self._unnormalize_x(batch[1:, ...])
        u_target, f_target = self.split_bundle(u_target * self.rescaler)
        steps = 0

        prediction_horizon = self.prediction_horizon * self.frame_stack
        n_context_frames = self.context_frames // self.frame_stack
        prediction_horizon = prediction_horizon // self.frame_stack
        open_loop_horizon = self.open_loop_horizon
        #assert open_loop_horizon <= prediction_horizon # for control only
        prediction_horizon = max(prediction_horizon, open_loop_horizon)

        u_mse_list = []
        f_mse_list = []
        u_trajectory = u_init[None]
        u_pred_trajectory = u_init[None]
        while steps < self.episode_len:
            # add noise to target_step
            """
            x: torch.Tensor - The current noisy sample tensor. This is the input that needs to be denoised.
            external_cond: Optional[torch.Tensor] - Optional external conditioning information (like goal states, constraints, etc.) that guides the generation process.
            curr_noise_level: torch.Tensor - The current noise level (timestep) in the diffusion process. This indicates how much noise is currently present.
            next_noise_level: torch.Tensor - The target noise level for the next step. The method will denoise from curr_noise_level to next_noise_level.
            guidance_fn: Optional[Callable] = None - An optional function that provides additional guidance during sampling (e.g., goal-directed guidance for planning tasks).
            """
            #state_hist = self.plan_pde(u_init, self.episode_len - steps, conditions)
            t_real = steps + n_context_frames
            #end = min(t_real + prediction_horizon + open_loop_horizon - 1, self.episode_len + n_context_frames)
            end = min(t_real + prediction_horizon, self.episode_len + n_context_frames)
            #prediction_window = [i for i in range(t_real, end)]
            #u_init = u_trajectory[-1:]
            #u_context = u_trajectory[-n_context_frames:]
            u_context = u_init[None]
            conditions_to_go = conditions[steps:end]

            #if steps % open_loop_horizon == 0:
            #state = self.plan_pde(u_init[None], self.episode_len - steps, conditions[steps:])
            state = self.plan_pde(u_context, end-t_real, conditions_to_go)
            state = self._unnormalize_x(state)
            #state_hist = self._unnormalize_x(state_hist)
            #state = state_hist[-1]

            # rescale for burgers numerical solver
            observation, control = self.split_bundle(state * self.rescaler)

            for i in range(open_loop_horizon):
                u_init = self._unnormalize_x(u_init)
                u_init, _ = self.split_bundle(u_init * self.rescaler)
                #u_init_debug = u_init.clone()
                # u_init shape (batch_size, observation_dim), control shape (frames, batch_size, control_dim)
                u_controlled = burgers_numeric_solve_free(
                    u_init,
                    #f_target[steps, :].unsqueeze(1),
                    control[i,:].unsqueeze(1),
                    visc=0.01,
                    T=0.1,
                    dt=1e-4,
                    num_t=1
                )
                u_mse = (u_controlled[:, -1, :] - u_target[steps, :]).square().mean(-1)  # shape: (batch_size,)
                f_mse = (control[i, :] - f_target[steps, :]).square().mean(-1)
                u_mse_list.append(u_mse.detach().cpu())
                f_mse_list.append(f_mse.detach().cpu())
                #loss = mse.mean()

                # Use the output of u_controlled to construct a new u_init for the next iteration
                # Concatenate u_controlled (last state) and force to form the new bundle
                new_u = u_controlled[:, -1, :] / self.rescaler  # de-scale to original range
                dummy_force = torch.zeros_like(new_u) # shape: (batch_size, control_dim)
                # update u init
                u_init = self._normalize_x(self.make_bundle(new_u, dummy_force))
                # record trajectory
                #u_trajectory = torch.cat([u_trajectory, u_init[None]], dim=0)
                steps += 1
                if steps >= self.episode_len:
                    break
            # import pdb; pdb.set_trace()

        # Compute average MSE over all steps
        u_mse_all = torch.stack(u_mse_list, dim=0)  # shape: (episode_len, batch_size)
        f_mse_all = torch.stack(f_mse_list, dim=0)


        self.log(f"{namespace}/u_rmse", u_mse_all.mean().sqrt().item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{namespace}/f_rmse", f_mse_all.mean().sqrt().item(), on_step=False, on_epoch=True, sync_dist=True)

        title = "u_rmse_per_step"
        self.log_line_plot(f"{namespace}/{title}", u_mse_all.mean(dim=-1).sqrt().cpu().numpy(),
                           "step", "rmse", title)
        title = "f_rmse_per_step"
        self.log_line_plot(f"{namespace}/{title}", f_mse_all.mean(dim=-1).sqrt().cpu().numpy(),
                           "step", "rmse", title)





    def plan_pde(self, init_state: torch.Tensor, horizon: int, conditions: Optional[Any] = None):
        def goal_guidance(x):

            # x is a tensor of shape [t b (fs c)]
            # pred shape [frame, batch, observation_dim + control_dim]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            # conditions shape [frame, batch, observation_dim]
            observation_pred, control_pred = self.split_bundle(x)
            observation_gt = conditions.detach().clone()

            weight = np.array([10] * self.frame_stack + [0.997**j for j in range(horizon)] + [0] * (h_padded - horizon))
            weight = torch.from_numpy(weight).float().to(self.device)

            dist = nn.functional.mse_loss(observation_pred, observation_gt, reduction="none")
            control_reg = control_pred ** 2

            weight = repeat(weight, "t -> t c", c=dist.shape[-1])

            episode_return = -(dist * weight[:, None] + 0.5 * control_reg).mean() * 1000

            # weight[self.frame_stack :, 1:] = 8
            # weight[: self.frame_stack, 1:] = 2

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None
        batch_size = init_state.shape[1]

        # plan_tokens: number of tokens (frames) for the plan
        #plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        #pad_tokens = self.n_tokens - plan_tokens - 1
        n_context = init_state.size(0)
        import pdb; pdb.set_trace()
        plan_window = [n_context+i for i in range(horizon)]
        plan, _ = self.model.model_sampling(xs_context=init_state, conditions=conditions, prediction_window=plan_window,
                                         n_frames=self.n_frames, batch_size=batch_size, guidance_fn=guidance_fn)

        # state_hist shape: (schedules, batch_size, observation_dim(u,f)) [65, 16, 50, 256]
        plan = rearrange(plan, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
        return plan
        

    def split_bundle(self, bundle):
        return torch.split(bundle, [self.observation_dim, self.control_dim], -1)

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if control is not None and valid_value is not None:
            valid_value = control
        if valid_value is None:
            raise ValueError("At least one of obs, control, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if control is None:
            control = torch.zeros(batch_shape + (self.control_dim,)).to(valid_value)


        bundle = [obs, control]

        return torch.cat(bundle, -1)

