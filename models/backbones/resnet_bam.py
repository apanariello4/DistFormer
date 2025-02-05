import argparse

import torch
import torch.nn as nn

from models.backbones.bam import BAM
from models.backbones.resnet import ResNet


class ResNetBAM(ResNet):
    def __init__(self, args: argparse.Namespace, depth: int):
        self.bam_centers = args.bam_centers
        super().__init__(args, depth)

        if not self.bam_centers:
            self.MEAN = self.MEAN[:3]
            self.STD = self.STD[:3]

    def perform_surgery(self):
        if self.bam_centers:
            conv1 = nn.Conv2d(
                in_channels=4,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            conv1.weight.data[:, :3, :, :] = self.resnet.conv1.weight.data

            self.resnet.conv1 = conv1

        def get_channels(n_channels, layer):
            if (layer == 1 and self.use_centers) or (
                layer != 1 and self.residual_centers
            ):
                return n_channels + 1
            return n_channels

        for i in range(1, 4):
            layer = getattr(self.resnet, f"layer{i}")
            last_conv = getattr(
                layer[-1], [c for c in dir(layer[-1]) if "conv" in c][-1]
            )

            bam = BAM(get_channels(last_conv.out_channels, i), last_conv.out_channels)
            setattr(self.resnet, f"layer{i}_bam", bam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input: (batch, channels, clip_len height, width),
        output: (batch, channels, clip_len height, width),"""

        if self.use_centers:
            res_init = x[:, -1].unsqueeze(1)
            if not self.bam_centers:
                x = x[:, :-1]

        if self.pretrained:
            x = self.normalize_input(x)
        # stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for i in range(1, self.last_layer + 1):
            layer = getattr(self.resnet, f"layer{i}")
            x = layer(x)
            if (
                i == 1
                and self.use_centers
                or (i not in (1, self.last_layer) and self.residual_centers)
            ):
                res = nn.functional.interpolate(
                    res_init, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, res], dim=1)

            if i != self.last_layer:
                x = getattr(self.resnet, f"layer{i}_bam")(x)

        return x


if __name__ == "__main__":
    from main import parse_args

    args = parse_args()
    args.backbone = "resnet18"
    args.pretrain = True
    args.shallow = False
    args.use_centers = True
    args.residual_centers = True
    model = ResNetBAM(args, 18)
