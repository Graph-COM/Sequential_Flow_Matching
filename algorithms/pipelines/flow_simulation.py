from typing import Optional, Any
from omegaconf import DictConfig
from ..base_task.simulation_task import SimulationTask
from ..base_model.flow_base import FlowMatchingBase

class FlowMatchingSimulation(SimulationTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingBase(cfg)

