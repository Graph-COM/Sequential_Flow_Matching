from typing import Optional, Any
from omegaconf import DictConfig
from ..base_task.simulation_task import SimulationTask
from ..base_model.autoregressive import Autoregressive

class AutoregressiveSimulation(SimulationTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = Autoregressive(cfg)

