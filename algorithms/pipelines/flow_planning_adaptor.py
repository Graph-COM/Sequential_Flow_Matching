from omegaconf import DictConfig
from ..base_model.flow_base_adaptor import FlowMatchingAdaptor
from ..base_task.maze_adaptation_task import MazeAdaptionTask

class FlowMatchingPlanningFineTuner(MazeAdaptionTask):
    """
    A fine-tuning version of FlowMatchingMaze that loads a pretrained checkpoint
    and creates a copy for fine-tuning while keeping the original model intact.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingAdaptor(cfg)