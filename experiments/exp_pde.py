from datasets.pde import BurgersDataset
from datasets.pde import FinetuningBurgersDataset
from datasets.smoke import SmokeDataset
from datasets.smoke import FinetuningSmokeDataset
from algorithms.pipelines import (DiffusionForcingPDE, FlowMatchingPDE, DiffusionForcingPDE2D, 
                            FlowMatchingPDE2D, DiffusionForcingPDE2DFineTuner, MeanFlowPDE2D, ConsistencyModelPDE2D,
                                  FlowMatchingPDE2DFineTuner)
from .exp_base import BaseLightningExperiment


class PDEExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_pde=DiffusionForcingPDE,
        df_pde_2D=DiffusionForcingPDE2D,
        df_pde_2D_finetuner=DiffusionForcingPDE2DFineTuner,
        flow_pde_2D_finetuner=FlowMatchingPDE2DFineTuner,
        flow_pde=FlowMatchingPDE,
        flow_pde_2D=FlowMatchingPDE2D,
        meanflow_pde_2D=MeanFlowPDE2D,
        cm_pde_2D=ConsistencyModelPDE2D,
    )

    compatible_datasets = dict(
        pde_burgers=BurgersDataset,
        pde_finetuning=FinetuningBurgersDataset,
        pde_smoke=SmokeDataset,
        pde_smoke_finetuning=FinetuningSmokeDataset,
    )