from einops.layers.torch import Rearrange
import torch.nn as nn

from models.temporal_compression.base_module import TemporalCompression


class TemporalConv(TemporalCompression):
    def __init__(self, input_size, clip_len, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.clip_len = clip_len
        self.conv = nn.Sequential(
            Rearrange("b t c h w -> b (t c) h w"),
            nn.Conv2d(
                self.input_size * self.clip_len,
                self.input_size,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        """input: (B, T, C, H, W) tensor
        output: (B, C, H, W) tensor"""
        return self.conv(x)

    @property
    def output_size(self):
        return self.input_size
