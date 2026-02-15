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
from ..base_task.simulation_task import SimulationTask
from ..base_task.simulation_2D_task import Simulation2DTask
from ..base_model.df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
)
from datasets.pde.solver import burgers_numeric_solve_free
# Run the following command to train and validate pde dataset:
# python -m main +name=pde_test experiment=exp_simulation algorithm=df_simulation dataset=simulation_burgers


class DiffusionForcingSimulation(SimulationTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)

class DiffusionForcingSimulation2D(Simulation2DTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)