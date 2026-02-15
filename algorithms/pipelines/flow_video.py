from omegaconf import DictConfig
from ..base_task.video_task import VideoTask
from ..base_model.flow_base import FlowMatchingBase

class FlowMatchingVideo(VideoTask):
    """
    A video prediction algorithm using Flow Matching.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingBase(cfg)