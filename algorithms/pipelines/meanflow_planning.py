from typing import Optional, Any
from omegaconf import DictConfig
from ..base_model.meanflow_base import MeanFlowBase
from ..base_task.maze_task import MazeTask

class MeanFlowMaze(MazeTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = MeanFlowBase(cfg)
