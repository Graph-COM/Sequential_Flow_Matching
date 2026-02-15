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

from ..base_task.tracking_task import TrackingTask
from ..base_model.cm_base import ConsistencyModelBase

class ConsistencyModelTracking(TrackingTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = ConsistencyModelBase(cfg)

    def load_model_weights_only(self, checkpoint_path):
        """
                Alternative method to load only model weights, bypassing PyTorch Lightning's checkpoint loading.
                Use this instead of trainer.fit() with resume_from_checkpoint for fine-tuning.
                """
        print(f"Loading model weights only from {checkpoint_path}")

        # Load checkpoint manually
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Create new state dict for fine-tuning structure
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.removeprefix('model.')
            new_state_dict[new_key] = value
        self.model.load_model_weights_only(new_state_dict)
        return True
