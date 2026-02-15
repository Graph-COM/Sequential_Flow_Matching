from omegaconf import DictConfig
from ..base_task.video_task import VideoTask
from ..base_model.df_base import DiffusionForcingBase


class DiffusionForcingVideo(VideoTask):
    """
    A video prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)