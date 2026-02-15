from omegaconf import DictConfig
from ..base_model.flow_base_adaptor import FlowMatchingAdaptor
from ..base_task.simulation_adaptation_task import SimulationAdaptionTask
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn.functional as F

class FlowMatchingSimulationFineTuner(SimulationAdaptionTask):
    """
    A fine-tuning version of FlowMatchingVideo that loads a pretrained checkpoint
    and creates a copy for fine-tuning while keeping the original model intact.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = FlowMatchingAdaptor(cfg)