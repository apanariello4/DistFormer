import argparse

import torch
import torch.nn as nn
import torchvision

from models.base_model import BaseModel

SHUFFLE_VARIANTS = {
    "shufflenet_v2_x0_5": torchvision.models.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": torchvision.models.shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": torchvision.models.shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": torchvision.models.shufflenet_v2_x2_0,
}


class ShuffleNet(BaseModel):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.pretrained = args.pretrain
        self.use_centers = args.use_centers
        self.residual_centers = args.residual_centers
        if self.residual_centers:
            print("Residual centers are not supported for ShuffleNet. Turning off.")
        self.net = SHUFFLE_VARIANTS[args.backbone](
            weights="DEFAULT" if self.pretrained else None
        )
        if self.pretrained:
            print(f"Using pretrained weights for {args.backbone}")

        self.perform_surgery()
        # get output channels of last conv layer
        self.output_size = self.net.conv5[0].out_channels
        self.scale = 1 / 32
        self.frame_size = self.get_frame_size(args)
        if self.use_centers:
            self.MEAN += [0]
            self.STD += [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = self.normalize_input(x)
        x = self.net.conv1(x)
        x = self.net.maxpool(x)
        x = self.net.stage2(x)
        x = self.net.stage3(x)
        x = self.net.stage4(x)
        x = self.net.conv5(x)
        return x

    def perform_surgery(self):
        self.net.fc = nn.Identity()

        if self.use_centers:
            old_conv = self.net.conv1[0]
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
            self.net.conv1[0] = new_conv


if __name__ == "__main__":
    args = argparse.Namespace()
    args.backbone = "shufflenet_v2_x1_5"
    args.pretrain = True
    args.shallow = False
    args.use_centers = False
    args.residual_centers = False
    model = ShuffleNet(args)
    print(model)
