import argparse
from einops import rearrange

import torch
import torch.nn as nn

from models.backbones.convnext_v2 import CustomConvNeXtV2
from models.backbones.resnet_fpn import TopDownPathway, get_img_size
from models.temporal_compression.biconvlstm import ConvLSTM


def get_output_channels(conv: nn.Module) -> int:
    return conv.out_channels


class ConvNeXtV2FPN(CustomConvNeXtV2):
    """https://github.com/haofengac/MonoDepth-FPN-PyTorch"""

    def __init__(self, args: argparse.Namespace, size: str, red_factor: float = 0.5):
        super().__init__(args, size)
        self.hdim = args.fpn_hdim
        self.idx_lstm = args.fpn_idx_lstm
        self.clip_len = args.clip_len

        if args.shallow:
            print("Shallow FPN not implemented yet (probably not needed)")

        # inplanes = [self.convnext.downsample_layers[0][0].out_channels]
        inplanes = [
            layer[-1].out_channels for layer in self.convnext.downsample_layers[1:]
        ]
        inplanes.append(inplanes[-1])

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

        self.scale = 0.125
        self.frame_size = self.get_frame_size(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input: (batch, channels, clip_len height, width),
        output: (batch, channels, clip_len height, width),"""

        if self.pretrained:
            x = self.normalize_input(x)

        hiddens = []

        for i in range(4):
            x = self.convnext.downsample_layers[i](x)
            hiddens.append(x) if i > 0 else None
            x = self.convnext.stages[i](x)

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

    B, C, W, H = 3, 4, 360, 640
    x = torch.rand((B, C, W, H))
    print(x.shape)
    net = ConvNeXtV2FPN(main.parse_args(), "tiny")
    output = net(x)
    print(output.shape)
