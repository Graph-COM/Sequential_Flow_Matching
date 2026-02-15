from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image
from ema_pytorch import EMA
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from typing import Any, Union, Sequence, Optional
import lightning.pytorch as pl
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
)
import einops
from abc import ABC, abstractmethod

class AbstractTask(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()

    @abstractmethod
    def _preprocess_batch(self, batch):
        raise NotImplementedError('Please implement _preprocess_batch in the task class')
        # return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)
        weights = masks.float()


        if self.cfg.get('context_mask'):
            T, B = xs.size(0), xs.size(1)
            cut = torch.randint(low=1, high=T, size=(B,), device=xs.device)
            t_idx = torch.arange(T, device=xs.device).unsqueeze(1)  # [T, 1]
            context_masks = t_idx >= cut.unsqueeze(0)
        else:
            context_masks = None

        xs_pred, loss = self.model(xs, conditions, masks, context_masks=context_masks)
        loss = self.reweight_loss(loss, weights)
        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1:]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1:]
        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }
        return output_dict

    @torch.no_grad()
    @abstractmethod
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        raise NotImplementedError('Please implement validation_step in the task class')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.model.configure_optimizers()

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")
        return x

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.model.data_mean.ndim) + list(self.model.data_mean.shape)
        mean = self.model.data_mean.reshape(shape)
        std = self.model.data_std.reshape(shape)
        # Avoid division by zero for std==0 (leave those values unchanged)
        std_safe = std.clone()
        std_safe[std_safe == 0] = 1.0
        return (xs - mean) / std_safe

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.model.data_mean.ndim) + list(self.model.data_mean.shape)
        mean = self.model.data_mean.reshape(shape)
        std = self.model.data_std.reshape(shape)
        # If std is zero, output mean
        std_safe = std.clone()
        std_safe[std_safe == 0] = 1.0
        return xs * std_safe + mean

    def _normalize_x_with_mean(self, xs, mean, std):
        shape = [1] * (xs.ndim - mean.ndim) + list(mean.shape)
        mean = torch.tensor(mean.reshape(shape)).to(xs.device)
        std = torch.tensor(std.reshape(shape)).to(xs.device)
        return (xs - mean) / std

    def _unnormalize_x_with_mean(self, xs, mean, std):
        shape = [1] * (xs.ndim - mean.ndim) + list(mean.shape)
        mean = torch.tensor(mean.reshape(shape)).to(xs.device)
        std = torch.tensor(std.reshape(shape)).to(xs.device)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)

    def _unstack_and_unnormalize_with_mean(self, xs, mean, std):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x_with_mean(xs, mean, std)

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
                )
            loss = loss * weight

        #return loss.mean()
        return loss.mean() * weight.numel() / (weight!=0).sum()

    def log_video(
            self,
            key: str,
            video: Union[np.ndarray, torch.Tensor],
            mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
            std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
            fps: int = 12,
            format: str = "mp4",
    ):
        """
        Log video to wandb. WandbLogger in pytorch lightning does not support video logging yet, so we call wandb directly.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height, width) or in the form
                (batch, time, channel, height, width). The content must be be in 0-255 if under dtype uint8
                or [0, 1] otherwise.
            mean: optional, the mean to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            key: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()

        expand_shape = [1] * (len(video.shape) - 2) + [3, 1, 1]
        if std is not None:
            if isinstance(std, (float, int)):
                std = [std] * 3
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu().numpy()
            std = np.array(std).reshape(*expand_shape)
            video = video * std
        if mean is not None:
            if isinstance(mean, (float, int)):
                mean = [mean] * 3
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            mean = np.array(mean).reshape(*expand_shape)
            video = video + mean

        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        # Check if logger and experiment are available
        if self.logger is None or self.logger.experiment is None:
            print(f"Warning: Logger not available, skipping log_video for {key}")
            return
            
        self.logger.experiment.log(
            {
                key: wandb.Video(video, fps=fps, format=format),
            },
            step=self.global_step,
        )

    def log_stream_video(self, xs, xs_pred):
        for start in range(xs.shape[0]):
            #for lead in range(xs.shape[1]):
            key = 'x_start_%d' % (start)
            self.log_image(key, xs[start], caption=['start_%d_lead_%d' % (start, lead) for lead in range(xs.shape[1])])
            key = 'xhat_start_%d' % (start)
            self.log_image(key, xs_pred[start], caption=['start_%d_lead_%d' % (start, lead) for lead in range(xs.shape[1])])
            #self.log_image(key, xs_pred[start, lead])


    def log_image(
            self,
            key: str,
            image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
            mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
            std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
            **kwargs: Any,
    ):
        """
        Log image(s) using WandbLogger.
        Args:
            key: the name of the video.
            image: a single image or a batch of images. If a batch of images, the shape should be (batch, channel, height, width).
            mean: optional, the mean to unnormalize image tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize tensor, assuming unnormalized data is in [0, 1].
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        if isinstance(image, Image.Image):
            image = [image]
        elif len(image) and not isinstance(image[0], Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image[None]

            if image.shape[1] == 3:
                if image.shape[-1] == 3:
                    warnings.warn(f"Two channels in shape {image.shape} have size 3, assuming channel first.")
                image = einops.rearrange(image, "b c h w -> b h w c")

            if std is not None:
                if isinstance(std, (float, int)):
                    std = [std] * 3
                if isinstance(std, torch.Tensor):
                    std = std.detach().cpu().numpy()
                std = np.array(std)[None, None, None]
                image = image * std
            if mean is not None:
                if isinstance(mean, (float, int)):
                    mean = [mean] * 3
                if isinstance(mean, torch.Tensor):
                    mean = mean.detach().cpu().numpy()
                mean = np.array(mean)[None, None, None]
                image = image + mean

            if image.dtype != np.uint8:
                image = np.clip(image, a_min=0.0, a_max=1.0) * 255
                image = image.astype(np.uint8)
                image = [img for img in image]

        # Check if logger is available
        if self.logger is None:
            print(f"Warning: Logger not available, skipping log_image for {key}")
            return
            
        self.logger.log_image(key=key, images=image, **kwargs)

    def log_line_plot(
            self,
            key: str,
            vector: Union[np.ndarray, torch.Tensor, Sequence[float]],
            xlabel: str,
            ylabel: str,
            title: str,
            **kwargs: Any,
    ):
        """
        Log a line plot to wandb.
        Args:
            key: the name of the plot.
            vector: the vector to plot.
            xlabel: the label of the x-axis.
            ylabel: the label of the y-axis.
            title: the title of the plot.
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        # Check if logger and experiment are available
        if self.logger is None or self.logger.experiment is None:
            print(f"Warning: Logger not available, skipping log_line_plot for {key}")
            return
            
        table = wandb.Table(
            data=[[i, v] for i, v in enumerate(vector)],
            columns=[xlabel, ylabel]
        )

        # log using Lightning's wandb logger
        self.logger.experiment.log({
            key:
                wandb.plot.line(
                    table,
                    x=xlabel,
                    y=ylabel,
                    title=title,
                    **kwargs
                )
        })

    def log_3d_line_plot(
            self,
            key: str,
            data1: np.ndarray,
            label1: str,
            data2: np.ndarray,
            label2: str,
            title: str,
            xlabel: str = "X [m]",
            ylabel: str = "Y [m]",
            zlabel: str = "Z [m]",
    ):
        """
        Generates a static 3D line plot and logs it directly to wandb as an image.

        Args:
            self: The object instance with a wandb logger (e.g., a PyTorch Lightning module).
            key: The key to log the image under in the wandb dashboard.
            data1: An (N, 3) array for the first trajectory (e.g., EKF).
            label1: The legend label for the first trajectory.
            data2: An (M, 3) array for the second trajectory (e.g., Ground Truth).
            label2: The legend label for the second trajectory.
            title: The title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            zlabel: Label for the z-axis.
        """
        # Create a figure and a 3D subplot
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first trajectory (e.g., EKF)
        ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], label=label1, color="blue", marker='.', markersize=4)

        # Plot the second trajectory (e.g., Ground Truth)
        ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], label=label2, color="black", marker='.', markersize=4)

        # --- Customize the Plot ---
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        # --- Save plot to an in-memory buffer ---
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)  # Rewind the buffer to the beginning

        # --- Log the image from the buffer to wandb ---
        from PIL import Image
        pil_image = Image.open(buf)
        self.logger.experiment.log({
            key: wandb.Image(pil_image, caption=title)
        })

        # Close the figure to free up memory
        plt.close(fig)


    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def load_model_weights_only(self, checkpoint_path):
        """
        Load model weights from a checkpoint file, bypassing PyTorch Lightning's checkpoint loading.
        This is useful when you want to load weights without restoring the full training state.
        """
        print(f"Loading model weights only from {checkpoint_path}")
        
        # Load checkpoint manually
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Load the state dict (PyTorch Lightning modules handle the key mapping automatically)
        self.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully")
        return True
