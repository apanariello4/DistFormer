import argparse
import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.backbones.resnet import ResNet
from models.temporal_compression.biconvlstm import ConvLSTM


def conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)


def conv3x3(in_planes: int, out_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)


def get_output_channels(resnet_block: nn.Module) -> int:
    return [x for x in resnet_block[-1].children() if isinstance(x, nn.Conv2d)][
        -1
    ].out_channels


def get_img_size(original_size: Iterable[int], depth: float) -> tuple[int, int]:
    return tuple(math.ceil(x * 0.5**depth) for x in original_size)


class LateralConnection(nn.Module):
    def __init__(self, inplanes: int, outplanes: int):
        super(LateralConnection, self).__init__()
        self.latlayer = conv1x1(inplanes, outplanes)
        self.smooth = conv3x3(outplanes, outplanes)

    def forward(self, x, y):
        _, _, H, W = y.size()
        y = self.latlayer(y)
        y = F.interpolate(x, size=(H, W), mode="bilinear") + y
        y = self.smooth(y)
        return y


class LateralBlock(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, upscale_factor: int = None):
        super(LateralBlock, self).__init__()
        self.layer = self.get_aggnode(inplanes, outplanes)
        self.upsample = nn.Sequential()
        if upscale_factor is not None:
            assert isinstance(upscale_factor, int)
            self.upsample = self.get_upshuffle(outplanes, outplanes, upscale_factor)

    def get_aggnode(self, in_planes: int, out_planes: int):
        return nn.Sequential(
            conv3x3(in_planes, in_planes),
            nn.ReLU(),
            conv3x3(in_planes, out_planes),
            nn.ReLU(),
        )

    def get_upshuffle(self, in_planes: int, out_planes: int, upscale_factor: int):
        return nn.Sequential(
            conv3x3(in_planes, out_planes * (upscale_factor**2)),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.upsample(self.layer(x))


class TopDownPathway(nn.Module):
    def __init__(
        self,
        inplanes: List[int],
        upscale_factors: list[None],
        hdim: int,
        hdim_red_factor: float,
        img_size: tuple[int, int],
        lstm_layer: Optional[str] = None,
        clip_len: int = 1,
    ) -> None:
        super(TopDownPathway, self).__init__()
        assert len(inplanes) == len(upscale_factors) == 4

        # Top layer
        self.toplayer = conv1x1(inplanes[3], hdim)

        # Lateral layers
        self.latlayer1 = LateralConnection(inplanes[2], hdim)
        self.latlayer2 = LateralConnection(inplanes[1], hdim)
        self.latlayer3 = LateralConnection(inplanes[0], hdim)

        hdim2 = int(hdim * hdim_red_factor)

        # Aggregate layers
        self.latblock1 = LateralBlock(hdim, hdim2, upscale_factors[3])
        self.latblock2 = LateralBlock(hdim, hdim2, upscale_factors[2])
        self.latblock3 = LateralBlock(hdim, hdim2, upscale_factors[1])
        self.latblock4 = LateralBlock(hdim, hdim2, upscale_factors[0])

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)

        p4 = self.latlayer1(p5, c4)
        p3 = self.latlayer2(p4, c3)
        p2 = self.latlayer3(p3, c2)

        d5 = self.latblock1(p5)
        d4 = self.latblock2(p4)
        d3 = self.latblock3(p3)
        d2 = self.latblock4(p2)

        H, W = d2.shape[-2:]
        return torch.cat([F.interpolate(d, size=(H, W), mode="bilinear") for d in (d5, d4, d3, d2)], dim=1)


class ResNetFPN(ResNet):
    """https://github.com/haofengac/MonoDepth-FPN-PyTorch"""

    def __init__(self, args: argparse.Namespace, depth: int, red_factor: float = 0.5):
        super().__init__(args, depth)
        self.hdim = args.fpn_hdim
        self.idx_lstm = args.fpn_idx_lstm
        self.clip_len = args.clip_len

        if args.shallow:
            print("Shallow FPN not implemented yet (probably not needed)")

        inplanes = [
            get_output_channels(getattr(self.resnet, f"layer{i}")) for i in range(1, 5)
        ]

        img_size = 0
        if self.idx_lstm in ("c3", "c4", "c5"):
            assert args.clip_len > 1
            self.idx_lstm = ["c3", "c4", "c5"].index(self.idx_lstm) + 2
            img_size = get_img_size(args.input_h_w, depth=self.idx_lstm + 1)
            self.lstm = ConvLSTM(
                input_dim=inplanes[self.idx_lstm - 1],
                hidden_dim=inplanes[self.idx_lstm - 1],
                img_size=img_size,
                kernel_size=(3, 3),
            )

        self.topdownpath = TopDownPathway(
            inplanes,
            [None, 2, 4, 8],
            self.hdim,
            red_factor,
            img_size=img_size,
            lstm_layer=self.idx_lstm,
            clip_len=self.clip_len,
        )

        self.output_size = 4 * int(self.hdim * red_factor)

        self.scale = 0.25
        self.frame_size = self.get_frame_size(args)

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

        hiddens = []

        for i in range(1, 5):
            layer = getattr(self.resnet, f"layer{i}")
            if self.residual_centers and i != 1:
                res = nn.functional.interpolate(
                    res_init, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, res], dim=1)
            x = layer(x)
            hiddens.append(x)

        if hasattr(self, "lstm"):
            hiddens = [
                rearrange(h, "(b t) c h w -> b c t h w", t=self.clip_len)
                for h in hiddens
            ]
            hiddens[self.idx_lstm - 1] = rearrange(
                self.lstm(hiddens[self.idx_lstm - 1])[0], "b 1 c h w -> b c 1 h w"
            )
            hiddens = [h[:, :, -1] for h in hiddens]

        return self.topdownpath(*hiddens)


if __name__ == "__main__":
    import main

    B, C, W, H = 3, 3, 360, 640
    x = torch.rand((B, C, W, H))
    print(x.shape)
    net = ResNetFPN(main.parse_args(), 34)
    output = net(x)
    print(output.shape)
