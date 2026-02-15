from typing import Optional, Union
from omegaconf import DictConfig
import pathlib
from lightning.pytorch.loggers.wandb import WandbLogger

from .exp_base import BaseExperiment
from .exp_video import VideoPredictionExperiment
try: # TODO: fix this later
    from .exp_pde import PDEExperiment
except:
    PDEExperiment = None
from .exp_simulation import SimulationExperiment
from .exp_tracking import TrackingExperiment
from .exp_planning import PlanningExperiment
from .exp_weather import WeatherExperiment

# each key has to be a yaml file under '[project_root]/configurations/experiment' without .yaml suffix
exp_registry = dict(
    exp_video=VideoPredictionExperiment,
    exp_planning=PlanningExperiment,
    exp_pde=PDEExperiment,
    exp_simulation=SimulationExperiment,
    exp_tracking=TrackingExperiment,
    exp_weather=WeatherExperiment,
)


def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)
