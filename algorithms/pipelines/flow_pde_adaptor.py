from omegaconf import DictConfig
from ..base_model.flow_base_adaptor import FlowMatchingAdaptor
from ..base_task.control_2D_adaptation_task import Control2DAdaptionTask


class FlowMatchingPDE2DFineTuner(Control2DAdaptionTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingAdaptor(cfg)