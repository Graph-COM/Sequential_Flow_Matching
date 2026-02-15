from typing import Optional, Any
from omegaconf import DictConfig
from ..base_task.tracking_task import TrackingTask
from ..base_model.flow_base import FlowMatchingBase

class FlowMatchingTracking(TrackingTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingBase(cfg)

