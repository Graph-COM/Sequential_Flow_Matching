from omegaconf import DictConfig
from ..base_model.df_base import DiffusionForcingBase
from ..base_task.maze_task import MazeTask


class DiffusionForcingMaze(MazeTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = DiffusionForcingBase(cfg)
