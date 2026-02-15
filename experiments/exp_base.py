"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict
import pathlib
import os

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero

from torch.utils.data._utils.collate import default_collate
from utils.dataloder_utils import customized_collate_fn

from omegaconf import OmegaConf

torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        self.root_cfg.algorithm.data_save_dir = self.root_cfg.dataset.save_dir # in case we need the dataset dir for the algorithm
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                #num_workers=4,
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        #shuffle = (
            #False
            #if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            #else self.cfg.validation.data.shuffle
        #)
        shuffle = self.cfg.validation.data.shuffle
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                #num_workers=4,
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_finetuning_loader(self, save_inference=False) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("finetuning", save_inference=save_inference)
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        #shuffle = shuffle if not save_inference else False
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                #num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                num_workers=4,
                shuffle=shuffle,
                persistent_workers=True,
                collate_fn = default_collate if save_inference else customized_collate_fn,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            #devices="auto",
            devices=1,
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            log_every_n_steps=1,  # Log every step
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        # in case we start with a pretrained model weight
        if self.ckpt_path:
            # Load weights manually to avoid checkpoint state issues
            self.algo.load_model_weights_only(self.ckpt_path)

        trainer.fit(
            self.algo,
            train_dataloaders=self._build_training_loader(),
            val_dataloaders=self._build_validation_loader(),
            #ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            #devices="auto",
            devices=1,
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            #devices="auto",
            devices=1,
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
        )

        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )


    def test_overfit(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            #devices="auto",
            devices=1,
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        trainer.validate(
            self.algo,
            dataloaders=self._build_training_loader(),
            ckpt_path=self.ckpt_path,
        )


    def save_inference(self) -> None:
        """
        run validation on the training dataset to get and save the inference results
        """
        from omegaconf import open_dict
        with open_dict(self.root_cfg.algorithm):
            self.root_cfg.algorithm.save_inference = True
            self.root_cfg.algorithm.ckpt_path = self.ckpt_path
        #self.root_cfg.algorithm = OmegaConf.create({**self.root_cfg.algorithm, "save_inference": True})
        #self.root_cfg.algorithm = OmegaConf.create({**self.root_cfg.algorithm, "ckpt_path": self.ckpt_path})

        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            #devices="auto",
            devices=1,
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        if self.ckpt_path:
            # Load weights manually to avoid checkpoint state issues
            self.algo.load_model_weights_only(self.ckpt_path)
        else:
            raise Exception('Please provide a model checkpoint path for running finetuning.')
        # run inference on training dataset
        trainer.validate(
            self.algo,
            dataloaders=self._build_finetuning_loader(save_inference=True),
        )

    def test_finetune_overfit(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            #devices="auto",
            devices=1,
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            dataloaders=self._build_finetuning_loader(save_inference=True),
            ckpt_path=self.ckpt_path,
        )

    def fine_tune(self) -> None:
        # call the self._build_fine_tuning_loader, which fetch the saved inference_results.pt
        # call the self.algo, whose train_step implements a fine-tuning of pretrained model, and validation_step
        # implements pretrain model + adaptor hybrid inference mode
        """
               All training happens here
               """
        # hardcode the ckpt
        from omegaconf import open_dict
        with open_dict(self.root_cfg.algorithm):
            self.root_cfg.algorithm.ckpt_path = self.ckpt_path
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            devices=1,
            #devices="auto",
            num_nodes=self.cfg.num_nodes,
            #strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            log_every_n_steps=1,  # Log every step
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")
        #trainer.fit(
            #self.algo,
            #train_dataloaders=self._build_finetuning_loader(),
            #val_dataloaders=self._build_validation_loader(),
            #ckpt_path=self.ckpt_path,
        #)
        #return

        # Check if we're in fine-tuning mode and need to load weights manually
        if self.ckpt_path:
            # Load weights manually to avoid checkpoint state issues
            self.algo.load_model_weights_only(self.ckpt_path)
            # Start training without checkpoint resumption
        #else:
            #raise Exception('Please provide a model checkpoint path for running finetuning.')

        trainer.fit(
            self.algo,
            train_dataloaders=self._build_finetuning_loader(),
            val_dataloaders=self._build_validation_loader(),
        )



    def _build_dataset(self, split: str, save_inference=False) -> Optional[torch.utils.data.Dataset]:
        if split in ["training", "test", "validation"]:
           return self.compatible_datasets[self.root_cfg.dataset._name](self.root_cfg.dataset, split=split)
        elif split == "finetuning":
            if save_inference:
                # we are running pretrained model on a specific dataset
                return self.compatible_datasets[self.root_cfg.dataset._name](self.root_cfg.dataset, split='finetune')
            else:
                # we are finetuning models based on the saved inference results
                return self.compatible_datasets[self.root_cfg.dataset._name + '_finetuning'](self.root_cfg.dataset,
                                                                alg_cfg=self.root_cfg.algorithm,)
                                                                #update_strategy=self.root_cfg.algorithm.update_strategy)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")

    def fine_tune_overfit(self) -> None:
        # call the self._build_fine_tuning_loader, which fetch the saved inference_results.pt
        # call the self.algo, whose train_step implements a fine-tuning of pretrained model, and validation_step
        # implements pretrain model + adaptor hybrid inference mode
        """
               All training happens here
               """
        # hardcode the ckpt
        from omegaconf import open_dict
        with open_dict(self.root_cfg.algorithm):
            self.root_cfg.algorithm.ckpt_path = self.ckpt_path
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            devices=1,
            # devices="auto",
            num_nodes=self.cfg.num_nodes,
            # strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            strategy='auto',
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            log_every_n_steps=1,  # Log every step
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")
        #trainer.fit(
        #self.algo,
        #train_dataloaders=self._build_finetuning_loader(),
        #val_dataloaders=self._build_finetuning_loader(save_inference=True),
        #ckpt_path=self.ckpt_path,
        #)
        #return

        # Check if we're in fine-tuning mode and need to load weights manually
        if self.ckpt_path:
            # Load weights manually to avoid checkpoint state issues
            self.algo.load_model_weights_only(self.ckpt_path)
            # Start training without checkpoint resumption
        #else:
            #raise Exception('Please provide a model checkpoint path for running finetuning.')

        trainer.fit(
            self.algo,
            train_dataloaders=self._build_finetuning_loader(),
            val_dataloaders=self._build_finetuning_loader(save_inference=True),
        )