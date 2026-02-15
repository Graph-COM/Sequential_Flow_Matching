from datasets import FinetuningVideoDataset
from datasets.video import (
    MinecraftVideoDataset,
    DmlabVideoDataset,
    FinetuningVideoDataset,
)
from algorithms.pipelines import (DiffusionForcingVideo, DiffusionForcingVideoFineTuner,
                                  FlowMatchingVideo, FlowMatchingVideoFineTuner)
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
        df_video_finetuner=DiffusionForcingVideoFineTuner,
        flow_video=FlowMatchingVideo,
        flow_video_finetuner=FlowMatchingVideoFineTuner,
    )

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
        video_dmlab=DmlabVideoDataset,
        video_minecraft_finetuning=FinetuningVideoDataset,
        video_dmlab_finetuning=FinetuningVideoDataset,
    )
