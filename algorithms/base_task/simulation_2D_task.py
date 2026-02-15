from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from ..common.abstract_task import AbstractTask
from omegaconf import OmegaConf

class Simulation2DTask(AbstractTask):
    def __init__(self, cfg: DictConfig):
        self.observation_dim = cfg.observation_shape
        self.control_dim = cfg.control_shape
        self.episode_len = cfg.episode_len

        self.observation_mean = cfg.observation_mean
        self.observation_std = cfg.observation_std
        self.control_mean = cfg.control_mean
        self.control_std = cfg.control_std
        self.rescaler = torch.tensor(cfg.rescaler).reshape(1, 1, 6, 1, 1)

        self.sliding_window = cfg.sliding_window
        self.open_loop_horizon = cfg.get('open_loop_horizon', 1)


        # Check below are needed?
        cfg.data_mean =self.observation_mean
        cfg.data_std = self.observation_std
        # mean = list(self.observation_mean)
        # std = list(self.observation_std)
        # cfg.data_mean = np.array(mean).tolist()
        # cfg.data_std = np.array(std).tolist()

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

        self.validation_step_outputs = []

        super().__init__(cfg)


    def _preprocess_batch(self, batch):
        # input example: [batch_sizes, nframes, types, width, height]
        batch, indices = batch[0], batch[1]
        
        # density, velocity_x, velocity_y
        observations = batch[:,:, :3] # shape: [batch_sizes, nframes, 3, width, height]
        batch_size, n_frames = observations.shape[:2]
        masks = torch.ones(n_frames, batch_size).to(observations.device)

        # check this do we need to remove the last control?
        # init_obs, obs = torch.split(observations, [1, n_frames - 1], dim=1)

        # # obs = self._normalize_x(obs)
        # # init_obs = self._normalize_x(init_obs[:, 0])

        # init_obs = self.pad_init(init_obs[:, 0], batch_first=True)
        # bundles = torch.cat([init_obs, obs], dim=1)
        observations = rearrange(observations, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        observations = observations.flatten(2, 3).contiguous()
        
        if self.cfg.external_cond_dim:
            controls = batch[:,:, -2:] # [batch_sizes, nframes, 2, width, height]
            smoke = batch[:,:, 3].unsqueeze(2) # [batch_sizes, nframes, 1, width, height]
            conditions = torch.cat((smoke, controls), dim=2)
            # controls = controls[..., : self.control_dim]
            # controls = controls[:, :-1]
            # init_controls = torch.zeros_like(controls[:, 0], device=controls.device).unsqueeze(1)
            # controls = torch.cat([init_controls, controls], dim=1)
            # # control_mean, control_std = self._load_data_mean_std(namespace="control")

            conditions = rearrange(conditions, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
            conditions = conditions.flatten(2, 3).contiguous()
        else:
            raise ValueError("Control conditions are required")

        # output shape: [n_frames, batch_size, types, width, height]
        return observations, conditions, masks

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
        print(f"loss: {loss}")

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

    def on_validation_epoch_end(self, namespace='') -> None:
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)

        # in case we need multiple runs of generation
        if self.cfg.get('num_gen_trials') is not None and self.cfg.num_gen_trials > 1:
            xs = xs.repeat_interleave(self.cfg.num_gen_trials, dim=1)
            conditions = conditions.repeat_interleave(self.cfg.num_gen_trials, dim=1) if conditions is not None else None

        _, batch_size, *_ = xs.shape
        #if self.guidance_scale == 0:
            #namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        if self.sliding_window:
            print("Running in sliding window mode for validation_step.")
            self.eval_simulation_sliding_window(xs, horizon, conditions, namespace)
        else:
            print("Running in autoregressive mode for validation_step.")
            self.eval_simulation_autoregressive(xs, horizon, conditions, namespace)

    def eval_simulation_sliding_window(self, batch: torch.Tensor, horizon=None, conditions=None, namespace="validation"):
        """
        Sliding window evaluation for simulation, similar to df_video.py validation_step.
        Uses ground truth context and re-forecasts periodically instead of autoregressive prediction.
        """
        # batch shape: (n_frames, batch_size, observation_dim)
        n_frames, batch_size, *_ = batch.shape
        self.rescaler = self.rescaler.to(batch.device)
        
        # Set up sliding window parameters
        prediction_horizon = self.prediction_horizon * self.frame_stack
        n_context_frames = self.context_frames // self.frame_stack
        prediction_horizon = prediction_horizon // self.frame_stack
        num_sliding_windows = n_frames - prediction_horizon - n_context_frames + 1
        open_loop_horizon = self.open_loop_horizon
        assert open_loop_horizon <= prediction_horizon

        # Initialize prediction storage
        xs_pred_best = torch.zeros_like(batch)
        xs_pred_at_ts = []
        xs_gt_at_ts = []
        mse_list = []
        # Sliding window loop
        for step in range(0, num_sliding_windows):
            t_real = n_context_frames + step
            end = min(t_real + prediction_horizon + open_loop_horizon - 1, n_frames)
            prediction_window = [i for i in range(t_real, end)]
            # print(prediction_window)

            # Re-forecast based on new ground-truth observations
            if step % open_loop_horizon == 0:
                # Ground-truth context before t_real is observed
                xs_pred_best[:t_real] = batch[:t_real].clone()
                
                # Generate predictions for the prediction window
                xs_pred_best[prediction_window], _ = self.model.model_sampling(
                    # when replanning, use ground-truth context or previous prediction
                    xs_context=batch[:t_real],
                    # xs_context=xs_pred_best[:t_real],
                    conditions=conditions[:t_real+len(prediction_window)], # use force before prediction end time
                    prediction_window=prediction_window,
                    n_frames=n_frames,
                    batch_size=batch_size
                )
            
            # fetch the best estimation so far
            xs_pred = xs_pred_best[t_real:t_real+prediction_horizon]* self.rescaler[:,:,:3]
            gt = batch[t_real:t_real+prediction_horizon] * self.rescaler[:,:,:3]
            xs_pred_at_ts.append(xs_pred.clone())
            xs_gt_at_ts.append(gt)
        
        # xs_pred_at_ts: (num_windows, pred_horizon, batch, obs_dim)
        xs_pred_at_ts = torch.stack(xs_pred_at_ts)
        xs_gt_at_ts = torch.stack(xs_gt_at_ts)        

        # TODO: this seems to only record one batch, which is currently fine for 1D burgers (only 100 data points).
        # (1) Global MSE: average over all sliding windows, prediction horizons, batch, obs_dim
        mse = (xs_pred_at_ts - xs_gt_at_ts).square()
        rmse = mse.sqrt()
        rmse_mean = rmse.mean().item()
        rmse_std = rmse.std().item()
        global_mse = mse.mean().item()
        self.log(f"{namespace}/mse_sliding_window", global_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{namespace}/rmse_sliding_window", rmse_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{namespace}/rmse_std_sliding_window", rmse_std, on_step=False, on_epoch=True, sync_dist=True)

        # (2) MSE on different prediction horizons: average over sliding windows, batch, obs_dim
        rmse_per_horizon = rmse.mean(dim=(0, 2, 3, 4, 5))  # shape: (pred_horizon,)
        title = "rmse_per_horizon"
        self.log_line_plot(f"{namespace}/{title}", rmse_per_horizon,
                        "prediction_horizon", "mse", title)

        # # (3) MSE on different sliding windows: average over pred_horizon, batch, obs_dim
        rmse_per_window = rmse.mean(dim=(1, 2, 3, 4, 5))  # shape: (num_windows,)
        title = "rmse_per_window"
        self.log_line_plot(f"{namespace}/{title}", rmse_per_window,
                        "window_index", "mse", title)

        

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

    def _load_data_mean_std(self, namespace: str = "control"):
        loaded_v = {}
        for k, v in [(f"mean", self.cfg.get(f"{namespace}_mean")), (f"std", self.cfg.get(f"{namespace}_std"))]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    loaded_v[k] = torch.from_numpy(np.load(v)).to(self.device)
                elif v.endswith(".pt"):
                    loaded_v[k] = torch.load(v).to(self.device)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                loaded_v[k] = torch.tensor(v).to(self.device)
        return loaded_v["mean"], loaded_v["std"]