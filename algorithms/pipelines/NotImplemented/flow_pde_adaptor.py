from omegaconf import DictConfig
from algorithms.flow_matching import FlowMatchingVideo
from algorithms.diffusion_adaptor.flow_base_adaptor import FlowMatchingAdaptor
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

class FlowMatchingVideoFineTuner(FlowMatchingAdaptor, FlowMatchingVideo):
    """
    A fine-tuning version of FlowMatchingVideo that loads a pretrained checkpoint
    and creates a copy for fine-tuning while keeping the original model intact.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)


    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        raise NotImplementedError('To be implemented')