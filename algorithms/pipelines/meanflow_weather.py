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

from ..base_task.weather_task import WeatherTask
from ..base_model.meanflow_base import MeanFlowBase

class MeanFlowWeather(WeatherTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = MeanFlowBase(cfg)
