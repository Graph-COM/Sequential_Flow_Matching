from omegaconf import DictConfig
from ..base_model.df_base import DiffusionForcingBase
from ..base_task.control_task import ControlTask
from ..base_task.control_2D_tasks import Control2DTask

class DiffusionForcingPDE(ControlTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)

class DiffusionForcingPDE2D(Control2DTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)
