import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class ChannelGate(nn.Module):
    def __init__(
        self,
        gate_channel: int,
        out_channel: int,
        reduction_ratio: int = 16,
        num_layers: int = 1,
    ):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module("flatten", nn.Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                f"gate_c_fc_{i}", nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            self.gate_c.add_module(
                f"gate_c_bn_{i+1}", nn.BatchNorm1d(gate_channels[i + 1])
            )
            self.gate_c.add_module(f"gate_c_relu_{i+1}", nn.ReLU(inplace=True))

        self.gate_c.add_module(
            "gate_c_fc_final", nn.Linear(gate_channels[-2], out_channel)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        avg_pool = F.avg_pool2d(x, kernel_size=(H, W), stride=(H, W))

        return repeat(self.gate_c(avg_pool), "b c -> b c h w", h=H, w=W)


class SpatialGate(nn.Module):
    def __init__(
        self,
        gate_channel: int,
        out_channel: int,
        reduction_ratio: int = 16,
        dilation_conv_num: int = 2,
        dilation_val: int = 4,
    ) -> None:
        super(SpatialGate, self).__init__()
        self.out_channel = out_channel
        reduced_channels = gate_channel // reduction_ratio
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, reduced_channels, kernel_size=1),
        )
        self.gate_s.add_module("gate_s_bn_reduce0", nn.BatchNorm2d(reduced_channels))
        self.gate_s.add_module("gate_s_relu_reduce0", nn.ReLU(inplace=True))
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                f"gate_s_conv_di_{i}",
                nn.Conv2d(
                    reduced_channels,
                    reduced_channels,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                f"gate_s_bn_di_{i}", nn.BatchNorm2d(reduced_channels)
            )
            self.gate_s.add_module(f"gate_s_relu_di_{i}", nn.ReLU(inplace=True))

        self.gate_s.add_module(
            "gate_s_conv_final", nn.Conv2d(reduced_channels, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return repeat(self.gate_s(x), "b 1 h w -> b c h w", c=self.out_channel)


class BAM(nn.Module):
    def __init__(self, gate_channel: int, out_channel: int):
        super(BAM, self).__init__()
        self.out_channel = out_channel
        self.channel_att = ChannelGate(gate_channel, out_channel)
        self.spatial_att = SpatialGate(gate_channel, out_channel)
        self.init_bam()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = 1 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x[:, : self.out_channel]

    def init_bam(self):
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    torch.nn.init.kaiming_normal_(
                        self.state_dict()[key], mode="fan_out"
                    )
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0


if __name__ == "__main__":
    bam = BAM(513, 512)
    print(bam)
    x = torch.randn(2, 513, 32, 32)
    y = bam(x)
    print(y.shape)
