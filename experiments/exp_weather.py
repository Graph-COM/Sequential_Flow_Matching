from datasets import BurgersDataset
try:
    from datasets.weather import WeatherDataset, FinetuningWeatherDataset
except ImportError:
    print("Warning: Weather datasets could not be imported due to missing dependencies (e.g. weatherbench2)")
    WeatherDataset, FinetuningWeatherDataset = None, None

try:
    from algorithms.pipelines import (FlowMatchingWeather, FlowMatchingWeatherFineTuner, DiffusionForcingWeather,
                                      DiffusionForcingWeatherFineTuner, AutoregressiveWeather, ConsistencyModelWeather,
                                      MeanFlowWeather)
except ImportError:
    print("Warning: Weather algorithms could not be imported due to missing dependencies")
    FlowMatchingWeather, FlowMatchingWeatherFineTuner, DiffusionForcingWeather, DiffusionForcingWeatherFineTuner = None, None, None, None
    
from .exp_base import BaseLightningExperiment


class WeatherExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        flow_weather=FlowMatchingWeather,
        flow_weather_finetuner=FlowMatchingWeatherFineTuner,
        df_weather=DiffusionForcingWeather,
        df_weather_finetuner=DiffusionForcingWeatherFineTuner,
        autoregressive_weather=AutoregressiveWeather,
        cm_weather=ConsistencyModelWeather,
        meanflow_weather=MeanFlowWeather,
    )

    compatible_datasets = dict(
        weather=WeatherDataset,
        weather_finetuning=FinetuningWeatherDataset,
    )