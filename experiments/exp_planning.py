from datasets import Maze2dOfflineRLDataset, FinetuningMaze2dDataset
from algorithms.pipelines import DiffusionForcingMaze, FlowMatchingMaze, DiffusionForcingPlanningFineTuner, FlowMatchingPlanningFineTuner, MeanFlowMaze
from .exp_base import BaseLightningExperiment


class PlanningExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_planning=DiffusionForcingMaze,
        flow_planning=FlowMatchingMaze,
        meanflow_planning=MeanFlowMaze,
        df_planning_finetuner=DiffusionForcingPlanningFineTuner,
        flow_planning_finetuner=FlowMatchingPlanningFineTuner,
    )

    compatible_datasets = dict(
        # Planning datasets
        maze2d_umaze=Maze2dOfflineRLDataset,
        maze2d_medium=Maze2dOfflineRLDataset,
        maze2d_large=Maze2dOfflineRLDataset,
        # Finetuning datasets
        maze2d_umaze_finetuning=FinetuningMaze2dDataset,
        maze2d_medium_finetuning=FinetuningMaze2dDataset,
        maze2d_large_finetuning=FinetuningMaze2dDataset,
    )
