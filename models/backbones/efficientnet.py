import argparse

import torch
import torch.nn as nn
import torchvision

from models.base_model import BaseModel

EN_VARIANTS = {
    "efficientnet-b0": torchvision.models.efficientnet_b0,
    "efficientnet-b1": torchvision.models.efficientnet_b1,
    "efficientnet-b2": torchvision.models.efficientnet_b2,
    "efficientnet-b3": torchvision.models.efficientnet_b3,
    "efficientnet-b4": torchvision.models.efficientnet_b4,
    "efficientnet-b5": torchvision.models.efficientnet_b5,
    "efficientnet-b6": torchvision.models.efficientnet_b6,
    "efficientnet-b7": torchvision.models.efficientnet_b7,
    "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
    "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
    "efficientnet_v2_l": torchvision.models.efficientnet_v2_l,
}


class EfficientNet(BaseModel):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.pretrained = args.pretrain
        self.use_centers = args.use_centers
        self.residual_centers = args.residual_centers
        if self.residual_centers:
            print("Residual centers are not supported for EfficientNet. Turning off.")
        self.net = EN_VARIANTS[args.backbone](
            weights="DEFAULT" if self.pretrained else None
        )
        if self.pretrained:
            print(f"Using pretrained weights for {args.backbone}")

        self.perform_surgery()
        # get output channels of last conv layer
        self.output_size = self.net.features[-1][0].out_channels
        self.scale = 1 / 32
        self.frame_size = self.get_frame_size(args)

        if self.use_centers:
            self.MEAN += [0]
            self.STD += [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = self.normalize_input(x)
        x = self.net.features(x)

        return x

    def perform_surgery(self):
        del self.net.classifier
        del self.net.avgpool

        if self.use_centers:
            old_conv = self.net.features[0][0]
            new_conv = nn.Conv2d(
                old_conv.in_channels + 1,
                old_conv.out_channels,
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                old_conv.dilation,
                old_conv.groups,
                old_conv.bias,
            )
            new_conv.weight.data[:, :-1, :, :] = old_conv.weight.data
            self.net.features[0][0] = new_conv


if __name__ == "__main__":
    args = argparse.Namespace()
    args.backbone = "efficientnet-b0"
    args.pretrain = True
    args.shallow = False
    args.use_centers = False
    args.residual_centers = False
    model = EfficientNet(args)
    print(model)
