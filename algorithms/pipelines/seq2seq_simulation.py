from typing import Optional, Any
from omegaconf import DictConfig
from ..base_task.simulation_task import SimulationTask
from ..base_model.seq2seq import Seq2Seq

class Seq2SeqSimulation(SimulationTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = Seq2Seq(cfg)

