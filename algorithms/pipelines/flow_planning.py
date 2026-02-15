from omegaconf import DictConfig
from ..base_model.flow_base import FlowMatchingBase
from ..base_task.maze_task import MazeTask


class FlowMatchingMaze(MazeTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingBase(cfg)