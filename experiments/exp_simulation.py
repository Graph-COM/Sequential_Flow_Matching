from datasets import BurgersDataset
from datasets.pde import FinetuningBurgersDataset
from datasets.smoke import SmokeDataset

from algorithms.pipelines import (FlowMatchingSimulation, MeanFlowSimulation, DiffusionForcingSimulation,
                                  FlowMatchingSimulationFineTuner, DiffusionForcingSimulationFineTuner,
                                  DiffusionForcingSimulation2D, Seq2SeqSimulation, AutoregressiveSimulation,
                                  ConsistencyModelSimulation)
from .exp_base import BaseLightningExperiment


class SimulationExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_simulation=DiffusionForcingSimulation,
        # To do: Merge 2D to 1D once both results are good
        df_simulation_2D=DiffusionForcingSimulation2D,
        flow_simulation=FlowMatchingSimulation,
        meanflow_simulation=MeanFlowSimulation,
        df_simulation_finetuner=DiffusionForcingSimulationFineTuner,
        flow_simulation_finetuner=FlowMatchingSimulationFineTuner,
        seq2seq_simulation=Seq2SeqSimulation,
        autoregressive_simulation=AutoregressiveSimulation,
        cm_simulation=ConsistencyModelSimulation,
    )

    compatible_datasets = dict(
        simulation_burgers=BurgersDataset,
        simulation_burgers_finetuning=FinetuningBurgersDataset,
        simulation_smoke=SmokeDataset,
    )