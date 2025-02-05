import argparse

import torch
import torch.nn as nn
import torchvision

from models.base_model import BaseModel

MOBILE_VARIANTS = {
    "mobilenet_v2": torchvision.models.mobilenet_v2,
    "mobilenet_v3_large": torchvision.models.mobilenet_v3_large,
    "mobilenet_v3_small": torchvision.models.mobilenet_v3_small,
}


class MobileNet(BaseModel):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.pretrained = args.pretrain
        self.use_centers = args.use_centers
        self.residual_centers = args.residual_centers
        if self.residual_centers:
            print("Residual centers are not supported for MobileNet. Turning off.")
        self.net = MOBILE_VARIANTS[args.backbone](
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
    args.backbone = "mobilenet_v2"
    args.pretrain = True
    args.use_centers = False
    args.residual_centers = False
    model = MobileNet(args)
    x = torch.rand((1, 3, 6, 736, 1280))
    y = model.forward(x)
    print(f"$> {args.backbone}")
