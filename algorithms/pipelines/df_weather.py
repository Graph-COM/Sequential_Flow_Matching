from omegaconf import DictConfig
from ..base_task.weather_task import WeatherTask
from ..base_model.df_base import DiffusionForcingBase

class DiffusionForcingWeather(WeatherTask):
    """
    A video prediction algorithm using Flow Matching.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)