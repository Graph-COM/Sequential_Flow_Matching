from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image
from ema_pytorch import EMA
from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.diffusion_forcing.df_base import DiffusionForcingBase
from algorithms.diffusion_forcing.df_pde import DiffusionForcingPDE
from algorithms.diffusion_adaptor.df_base_adaptor import DiffusionAdaptor
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
)
from datasets.pde.solver import burgers_numeric_solve_free

# Run the following command to train and validate pde dataset:
# python -m main +name=pde_test experiment=exp_pde algorithm=df_pde dataset=pde_burgers


class DiffusionForcingPDEFinetuner(DiffusionAdaptor, DiffusionForcingPDE):
    def __init__(self, cfg: DictConfig):

        # Initialize the parent class first
        super().__init__(cfg)

    def eval_pde(self, batch: torch.Tensor, horizon=None, conditions=None, namespace="validation"):
        raise NotImplementedError('Implementation not finished!')
        trajectory = []
        all_plan_hist = []
        # batch shape: (n_frames, batch_size, observation_dim(u,f))
        # u_init shape: (batch_size, observation_dim(u,f))
        u_init = batch[0, ...]
        # the state in batch[t:, ...] is the same as conditions[t]
        u_target = self._unnormalize_x(batch[1:, ...])
        u_target, _ = self.split_bundle(u_target * self.rescaler)
        steps = 0

        mse_list = []
        while steps < self.episode_len:
            # add noise to target_step
            """
            x: torch.Tensor - The current noisy sample tensor. This is the input that needs to be denoised.
            external_cond: Optional[torch.Tensor] - Optional external conditioning information (like goal states, constraints, etc.) that guides the generation process.
            curr_noise_level: torch.Tensor - The current noise level (timestep) in the diffusion process. This indicates how much noise is currently present.
            next_noise_level: torch.Tensor - The target noise level for the next step. The method will denoise from curr_noise_level to next_noise_level.
            guidance_fn: Optional[Callable] = None - An optional function that provides additional guidance during sampling (e.g., goal-directed guidance for planning tasks).
            """
            state_hist = self.plan_pde(u_init, self.episode_len - steps, conditions)
            state_hist = self._unnormalize_x(state_hist)
            state = state_hist[-1]

            # rescale for burgers numerical solver
            observation, control = self.split_bundle(state * self.rescaler)

            u_init = self._unnormalize_x(u_init)
            u_init, _ = self.split_bundle(u_init * self.rescaler)
            u_init_debug = u_init.clone()
            # u_init shape (batch_size, observation_dim), control shape (frames, batch_size, control_dim)
            u_controlled = burgers_numeric_solve_free(
                u_init,
                control[0, :].unsqueeze(1),
                visc=0.01,
                T=0.1,
                dt=1e-4,
                num_t=1
            )
            mse = (u_controlled[:, -1, :] - u_target[steps, :]).square().mean(-1)  # shape: (batch_size,)
            mse_list.append(mse.detach().cpu())
            loss = mse.mean()

            # Use the output of u_controlled to construct a new u_init for the next iteration
            # Concatenate u_controlled (last state) and force to form the new bundle
            new_u = u_controlled[:, -1, :] / self.rescaler  # de-scale to original range
            dummy_force = torch.zeros_like(new_u)  # shape: (batch_size, control_dim)
            u_init = self._normalize_x(self.make_bundle(new_u, dummy_force))
            steps += 1
            # import pdb; pdb.set_trace()

        # Compute average MSE over all steps
        mse_all = torch.stack(mse_list, dim=0)  # shape: (episode_len, batch_size)

        self.log(f"{namespace}/mse", mse_all.mean().item(), on_step=False, on_epoch=True, sync_dist=True)

