import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange

from models.backbones import BACKBONES
from models.base_model import BaseLifter
from models.bb_regressors import REGRESSORS
from models.losses.balanced_l1 import BalancedL1Loss
from models.losses.gaussian_nll import GNLL
from models.losses.laplacian_nll import LNLL
from models.losses.smooth_l1 import SmoothL1
from models.losses.zhu_enhanced import EnhancedZHU
from trainers.base_trainer import Trainer
from trainers.trainer_regressor import TrainerRegressor


class ZHU(BaseLifter):
    TEMPORAL = False

    def __init__(self, args: argparse.Namespace, enhanced: bool = False):
        super().__init__()
        args.use_centers = False
        self.enhanced = enhanced
        self.alpha = args.alpha_zhu
        self.loss = args.loss

        assert not (
            self.enhanced and self.loss in ("gaussian", "laplacian")
        ), "Enhanced ZHU is not compatible with gaussian or laplacian loss"

        self.output_size = 2 if self.loss in ("gaussian", "laplacian") else 1

        self.backbone = BACKBONES[args.backbone](args)
        self.regressor = REGRESSORS[args.regressor](
            input_dim=self.backbone.output_size,
            pool_size=2,
            scale=self.backbone.scale,
            roi_op=torchvision.ops.roi_pool,
        )

        self.distance_estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.output_size * 2 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
        )

        if self.enhanced:
            self.keypoint_regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.backbone.output_size * 2 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Tanh(),
            )

    def forward(self, x: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        W = x.shape[-1]
        x = rearrange(x, "b c 1 h w -> b c h w")
        x = self.backbone(x)

        x = rearrange(x, "b c h w -> b c 1 h w")

        x = self.regressor(x, bboxes, scale=x.shape[-1] / W)

        z = self.distance_estimator(x).squeeze(-1)
        if self.loss in ("gaussian", "laplacian"):
            mu = F.softplus(z[..., 0])
            logvar = z[..., 1]
            z = (mu, logvar)
        else:
            z = F.softplus(z)

        if self.enhanced and self.training:
            k = self.keypoint_regressor(x)
            return z, k

        return z

    def get_loss_fun(self, **kwargs) -> nn.Module:
        if self.enhanced:
            return EnhancedZHU(alpha=self.alpha, train=self.training)
        return {
            "l1": SmoothL1,
            "gaussian": GNLL,
            "laplacian": LNLL,
            "balanced_l1": BalancedL1Loss,
        }[self.loss]()

    def get_trainer(self) -> Trainer:
        return TrainerRegressor
