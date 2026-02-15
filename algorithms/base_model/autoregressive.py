"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Any
from einops import rearrange
from .models.unet3d import Unet3D
from .models.transformer import Transformer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from .models.utils import EinopsWrapper

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.flow import FlowMatching


class Autoregressive(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.prediction_horizon = cfg.get('prediction_horizon')
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim
        self.causal = cfg.causal
        self.numerical_stabilizer = cfg.flow.get('numerical_stabilizer')
        self.n_frames = cfg.get('n_frames') if cfg.get('n_frames') is not None else cfg.get('episode_len') + cfg.frame_stack
        self.n_tokens = self.n_frames // cfg.frame_stack

        self.validation_step_outputs = []
        super().__init__(cfg)

    def _build_model(self):
        flow_model = FlowMatching(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.flow,
        )
        self.model = flow_model.model
        del flow_model
        self.pad_emb = Parameter(torch.randn(self.x_stacked_shape))
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + [self.pad_emb]
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def forward(self, xs: torch.Tensor, conditions: torch.Tensor, masks: torch.Tensor, context_masks: torch.Tensor=None):
        T, B = xs.size(0), xs.size(1)
        # auto-regressive model does not support future conditions yet
        pred = self.model(xs, torch.zeros([T, B]).to(xs.device), None, is_causal=self.causal)

        loss = F.mse_loss(pred[:-1], xs.detach()[1:], reduction="none")
        loss = torch.cat([torch.zeros_like(loss)[0:1],loss], dim=0)
        return pred, loss

    def model_sampling(self, xs_context, conditions, prediction_window, n_frames, batch_size, diff_hist=False,
                       guidance_fn=None):
        prediction_horizon = len(prediction_window)
        #pad_expand = self.pad_emb[None, None]  # [1,1,D]
        #pad_expand = pad_expand.expand(prediction_horizon, batch_size, *pad_expand.shape[2:])  # [T,B,D]
        #xs = torch.cat([xs_context, pad_expand], dim=0)

        # auto-regressive model does not support future conditions yet
        xs = xs_context.clone()
        for i in range(prediction_horizon):
            pred = self.model(xs, torch.zeros([xs.size(0), batch_size]).to(xs.device), None, is_causal=self.causal)
            xs = torch.cat([xs, pred[-1:]], dim=0)
        xs = xs.float()
        return xs[prediction_window], None

    def training_step(self, batch, batch_idx):
        raise NotImplementedError('Implement training_step in task class')
