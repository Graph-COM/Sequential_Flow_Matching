from omegaconf import DictConfig
from ..base_model.df_base_adaptor import DiffusionAdaptor
from ..base_task.weather_adaptation_task import WeatherAdaptionTask
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn.functional as F

class DiffusionForcingWeatherFineTuner(WeatherAdaptionTask):
    """
    A fine-tuning version of FlowMatchingVideo that loads a pretrained checkpoint
    and creates a copy for fine-tuning while keeping the original model intact.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionAdaptor(cfg)