from datasets import SSMDataset, FinetuningSSMataset
from algorithms.pipelines import (FlowMatchingTracking, FlowMatchingTrackingFineTuner, DiffusionForcingTracking,
                                  DiffusionForcingTrackingFineTuner, ConsistencyModelTracking, MeanFlowTracking)
from .exp_base import BaseLightningExperiment


class TrackingExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        flow_tracking=FlowMatchingTracking,
        df_tracking=DiffusionForcingTracking,
        flow_tracking_finetuner=FlowMatchingTrackingFineTuner,
        df_tracking_finetuner=DiffusionForcingTrackingFineTuner,
        cm_tracking=ConsistencyModelTracking,
        meanflow_tracking=MeanFlowTracking,
    )

    compatible_datasets = dict(
        state_tracking=SSMDataset,
        state_tracking_finetuning=FinetuningSSMataset,
    )