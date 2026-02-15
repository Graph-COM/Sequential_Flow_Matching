from omegaconf import DictConfig
from ..base_model.flow_base import FlowMatchingBase
from ..base_task.control_task import ControlTask
from ..base_task.control_2D_tasks import Control2DTask


class FlowMatchingPDE(ControlTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingBase(cfg)

class FlowMatchingPDE2D(Control2DTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingBase(cfg)