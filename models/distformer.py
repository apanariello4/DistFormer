import argparse
from itertools import chain
from typing import Optional, Tuple, Type

import torch
import torchvision
from einops import rearrange
from torch import nn

from models.backbones import BACKBONES
from models.base_model import BaseLifter
from models.bb_regressors import REGRESSORS
from models.losses.balanced_l1 import BalancedL1Loss
from models.losses.gaussian_nll import GNLL
from models.losses.laplacian_nll import LNLL
from models.losses.mse import MSE
from models.losses.smooth_l1 import SmoothL1
from models.temporal_compression import TEMPORAL_COMPRESSORS
from trainers.trainer_regressor import TrainerRegressor
from utils.bboxes import resized_crops


class DistFormer(BaseLifter):
    TEMPORAL = True

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.clip_len = args.clip_len
        self.pool_size = args.pool_size
        self.loss = args.loss
        self.estimator_input = args.estimator_input
        self.transformer_dropout = args.transformer_dropout
        self.use_geom = args.use_geom
        self.roi_op = (torchvision.ops.roi_align if args.roi_op == "align" else torchvision.ops.roi_pool)

        output_size = 2 if self.loss in ("laplacian", "gaussian") else 1

        self.backbone = BACKBONES[args.backbone](args)
        if self.backbone.output_size > 256 and "fpn" not in args.backbone:
            self.projection = nn.Conv2d(
                self.backbone.output_size, 256, kernel_size=1, bias=False
            )
            temp_in_channels = 256
        else:
            self.projection = nn.Identity()
            temp_in_channels = self.backbone.output_size

        self.temporal_compression = TEMPORAL_COMPRESSORS[args.temporal_compression](
            input_size=temp_in_channels,
            backbone=self.backbone,
            clip_len=self.clip_len,
            args=args,
        )

        self.regressor = REGRESSORS[args.regressor](
            self.temporal_compression.output_size,
            output_dim=self.estimator_input,
            pool_size=self.pool_size,
            scale=self.backbone.scale,
            dropout=self.transformer_dropout,
            transformer_aggregate=args.transformer_aggregate,
            args=args,
            roi_op=self.roi_op,
            norm_pix_loss=args.norm_pix_loss,
            loss=args.mae_loss,
        )

        if self.use_geom:
            self.dist_estimator = nn.Linear(self.estimator_input * 2, output_size)

            self.geom_embedding = nn.Sequential(
                nn.Linear(4, 256),
                nn.ReLU(),
                nn.Linear(256, self.estimator_input),
                nn.ReLU(),
                nn.Linear(self.estimator_input, self.estimator_input),
            )
        else:
            self.dist_estimator = nn.Linear(self.estimator_input, output_size)

    def forward(
        self,
        x: torch.Tensor,
        bboxes: list,
        clip_clean: Optional[torch.Tensor] = None,
        save_crops: bool = False,
        gt_anchors: Optional[torch.Tensor] = None,
        dataset_anchor: torch.Tensor = None,
        do_mae: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x shape: (batch, channels, clip_len, height, width)"""
        B, _, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.backbone(x)

        if x.shape[0] == B * T:
            x = self.projection(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=T)

            x = self.temporal_compression(x)
        else:
            x = rearrange(x, "b c h w -> b c 1 h w")

        if self.args.regressor == "mae":
            crops_list, small_bboxes = None, None
            if self.args.mae_target == "input" and self.training:
                # resize each crop to the same size (args.mae_crop_size)
                crops_list, small_bboxes = resized_crops(
                    bboxes,
                    clip_clean.to(self.args.device),
                    self.args.mae_crop_size,
                    self.args.mae_min_bbox_w,
                    self.args.mae_min_bbox_h,
                )

            x = self.regressor(
                x,
                bboxes,
                scale=x.shape[-1] / W,
                crops=crops_list,
                save_crops=save_crops,
                small_bboxes=small_bboxes,
                gt_anchors=gt_anchors,
                encode_only=not do_mae,
            )
            if self.training:
                x, losses = x
        else:
            x = self.regressor(x, bboxes, scale=x.shape[-1] / W, return_features=False)

        if self.use_geom:
            all_bbox = (
                torch.cat(tuple(chain.from_iterables(bboxes)), dim=0)
                .type_as(x)
                .to(x.device)
            )
            all_bbox = all_bbox / torch.tensor(
                [W, H, W, H], dtype=all_bbox.dtype, device=all_bbox.device
            )
            if all_bbox.dim() == 3:
                all_bbox = all_bbox.squeeze(0)
            geom_emb = self.geom_embedding(all_bbox)
            x = torch.cat([x, geom_emb], dim=-1)

        x = self.dist_estimator(x)
        if self.loss in ("laplacian", "gaussian"):
            mean, logvar = rearrange(x, "b (split 1) -> split b", split=2)
            x = (mean, logvar)
        else:
            x = x.squeeze(-1)

        if self.args.regressor == "mae" and self.training:
            return x, losses

        if not self.training:
            return x

        return x

    def get_loss_fun(self, **kwargs) -> nn.Module:
        return {
            "gaussian": GNLL,
            "laplacian": LNLL,
            "mse": MSE,
            "l1": SmoothL1,
            "balanced_l1": BalancedL1Loss,
        }[self.loss](**kwargs)

    def get_trainer(self) -> Type[TrainerRegressor]:
        return TrainerRegressor


def main():
    from main import parse_args

    args = parse_args()

    model = DistFormer(args).to(args.device)
    x = torch.rand((args.batch_size, 4, 512, 1024)).to(args.device)
    y = model.forward(x)

    print(f"$> RESNET-{model.depth}")
    print(f"───$> input shape: {tuple(x.shape)}")
    print(f"───$> output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
