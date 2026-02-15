from typing import Optional, Any
from omegaconf import DictConfig
from ..base_model.meanflow_base import MeanFlowBase
from ..base_task.control_task import ControlTask
from ..base_task.control_2D_tasks import Control2DTask

class MeanFlowPDE2D(Control2DTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = MeanFlowBase(cfg)
