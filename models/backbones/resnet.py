import argparse
import torch
import torch.nn as nn
import torchvision

from models.base_model import BaseModel

RES_VARIANTS = {
    18: torchvision.models.resnet18,
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}

OUT_CHANNELS = {
    18: 512,
    34: 512,
    50: 2048,
    101: 2048,
    152: 2048,
}


class ResNet(BaseModel):
    def __init__(self, args: argparse.Namespace, depth: int):
        super().__init__()
        self.depth = depth
        self.pretrained = args.pretrain
        self.shallow = args.shallow
        self.last_layer = 3 if self.shallow else 4
        self.use_centers = args.use_centers
        self.residual_centers = args.residual_centers
        assert not (
            not self.use_centers and self.residual_centers
        ), "Cannot use residual centers without using centers"
        self.resnet = RES_VARIANTS[self.depth](
            weights="DEFAULT" if self.pretrained else None
        )
        if self.pretrained:
            print(f"Using ImageNet pretrained weights for ResNet_{depth}")

        if self.shallow:
            self.output_size = OUT_CHANNELS[self.depth] // 2
            self.scale = 1 / 16
        else:
            self.output_size = OUT_CHANNELS[self.depth]
            self.scale = 1 / 32

        self.frame_size = self.get_frame_size(args)

        if self.use_centers:
            self.perform_surgery()
            self.MEAN += [0]
            self.STD += [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input: (batch, channels, clip_len height, width),
        output: (batch, channels, clip_len height, width),"""

        if self.pretrained:
            x = self.normalize_input(x)

        if self.residual_centers:
            res_init = x[:, -1].unsqueeze(1)

        # stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for i in range(1, self.last_layer + 1):
            layer = getattr(self.resnet, f"layer{i}")
            if self.residual_centers and i != 1:
                res = nn.functional.interpolate(
                    res_init, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, res], dim=1)
            x = layer(x)

        return x

    def perform_surgery(self):
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

        if not self.residual_centers:
            return

        def get_params_from_conv(myconv):
            return (
                myconv.weight.shape[1],
                myconv.weight.shape[0],
                myconv.stride,
                myconv.padding,
                tuple(myconv.weight.shape[2:]),
            )

        for i in range(2, 5):
            layer = getattr(self.resnet, f"layer{i}")

            old_conv = layer[0].conv1
            old_in_ch, old_out_ch, old_stride, old_padding, old_kernel_size = (
                get_params_from_conv(old_conv)
            )

            new_conv = nn.Conv2d(
                in_channels=old_in_ch + 1,
                out_channels=old_out_ch,
                kernel_size=old_kernel_size,
                stride=old_stride,
                padding=old_padding,
                bias=False,
            )

            new_conv.weight.data[:, :old_in_ch, :, :] = old_conv.weight.data
            layer[0].conv1 = new_conv

            if i != 1 or self.depth >= 50:
                old_downsample = layer[0].downsample[0]
                old_in_ch, old_out_ch, old_stride, _, _ = get_params_from_conv(
                    old_downsample
                )
                new_downsample = nn.Conv2d(
                    in_channels=old_in_ch + 1,
                    out_channels=old_out_ch,
                    kernel_size=1,
                    stride=old_stride,
                    padding=0,
                    bias=False,
                )
                new_downsample.weight.data[:, :old_in_ch, :, :] = (
                    old_downsample.weight.data
                )
                layer[0].downsample[0] = new_downsample
