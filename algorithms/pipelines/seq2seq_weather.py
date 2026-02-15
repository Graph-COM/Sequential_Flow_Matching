from omegaconf import DictConfig
from ..base_task.weather_task import WeatherTask
from ..base_model.seq2seq import Seq2Seq

class Seq2SeqWeather(WeatherTask):
    """
    A video prediction algorithm using Flow Matching.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = Seq2Seq(cfg)