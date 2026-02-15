from typing import Optional, Any
from omegaconf import DictConfig
from ..base_task.weather_task import WeatherTask
from ..base_model.autoregressive import Autoregressive

class AutoregressiveWeather(WeatherTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = Autoregressive(cfg)

