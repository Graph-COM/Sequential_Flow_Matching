from omegaconf import DictConfig
from ..base_model.df_base_adaptor import DiffusionAdaptor
from ..base_task.control_2D_adaptation_task import Control2DAdaptionTask


class DiffusionForcingPDE2DFineTuner(Control2DAdaptionTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionAdaptor(cfg)