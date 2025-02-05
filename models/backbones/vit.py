import argparse

import torch
import torch.nn as nn
import torchvision
from einops import rearrange

from models.base_model import BaseModel


class ViT(BaseModel):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.pretrained = args.pretrain

        self.use_centers = args.use_centers

        self.backbone = torchvision.models.vit_b_16(
            weights="DEFAULT" if self.pretrained else None
        )
        if self.pretrained:
            print("Using ImageNet pretrained weights for vit_b_16")
        self.output_size = 768
        self.scale = 1 / 16
        self.frame_size = self.get_frame_size(args)

        if self.use_centers:
            self.perform_surgery()
            self.MEAN += [0]
            self.STD += [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input: (b t) c h w,
        output: (b t) c h w,"""

        if self.pretrained:
            x = self.normalize_input(x)

        x = self.backbone._process_input(x)
        n = x.shape[0]

        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        # cls bad
        x = x[:, 1:, :]

        x = rearrange(x, "b (h w) c -> b c h w", h=14)

        return x

    def perform_surgery(self):
        target_conv = self.backbone.conv_proj

        old_in_ch, old_out_ch, old_stride, old_padding, old_kernel_size = (
            self.get_params_from_conv(target_conv)
        )

        new_conv = nn.Conv2d(
            in_channels=old_in_ch + 1,
            out_channels=old_out_ch,
            kernel_size=old_kernel_size,
            stride=old_stride,
            padding=old_padding,
        )

        new_conv.weight.data[:, :3, :, :] = target_conv.weight.data

        self.backbone.conv_proj = new_conv
