import argparse
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from models.temporal_compression.base_module import TemporalCompression


class NLBlockND(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        mode="embedded",
        dimension=3,
        bn_layer=True,
        sub_sample=False,
    ):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm


        https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/non_local.py
        """
        super(NLBlockND, self).__init__()

        assert dimension in range(1, 4), "Dimension can only be 1, 2, or 3"

        if mode not in ("gaussian", "embedded", "dot", "concatenate"):
            raise ValueError(
                "`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`"
            )

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
        )

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                ),
                bn(self.in_channels),
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
            )

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode in ("embedded", "dot", "concatenate"):
            self.theta = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
            )
            self.phi = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
            )

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1
                ),
                nn.ReLU(),
            )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            if hasattr(self, "phi"):
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer
        else:
            if not hasattr(self, "phi"):
                self.phi = nn.Identity()

    def forward(self, x):
        """
        args
            x: (B, C, T, H, W) for dimension=3; (B, C, H, W) for dimension 2; (B, C, T) for dimension 1
        """

        batch_size = x.shape[0]

        # (B, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = rearrange(self.g(x), "b c ... -> b (...) c")

        if self.mode == "gaussian":
            theta_x = rearrange(x, "b c ... -> b (...) c")
            phi_x = rearrange(self.phi(x), "b c ... -> b c (...)")
            f = theta_x @ phi_x

        elif self.mode in ("embedded", "dot"):
            theta_x = rearrange(self.theta(x), "b c ... -> b (...) c")
            phi_x = rearrange(self.phi(x), "b c ... -> b c (...)")
            f = theta_x @ phi_x

        elif self.mode == "concatenate":
            theta_x = rearrange(self.theta(x), "b c ... -> b c (...) 1")
            phi_x = rearrange(self.phi(x), "b c ... -> b c 1 (...)")
            theta_x = repeat(theta_x, "b c n 1 -> b c n w", w=phi_x.size(3))
            phi_x = repeat(phi_x, "b c 1 m -> b c h m", h=theta_x.size(2))

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode in ("gaussian", "embedded"):
            f_div_C = F.softmax(f, dim=-1)
        else:
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = rearrange(f_div_C @ g_x, "b f c -> b c f")
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        return self.W_z(y) + x


class EfficientNLBlockND(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inter_channels: Optional[int] = None,
        mode: str = "embedded",
        dimension: int = 3,
        bn_layer: bool = True,
        sub_sample: bool = False,
        phi_ksize: int = 1,
    ) -> None:
        """Implementation of Non-Local Block with 4 different pairwise functions
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
            sub_sample: whether to use subsampling trick
            phi_ksize: kernel size for phi convolution

        https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/non_local.py
        """
        super().__init__()

        assert dimension == 3 and phi_ksize in (1, 3)

        if mode not in ("gaussian", "embedded", "dot"):
            raise ValueError("`mode` must be one of `gaussian`, `embedded`, `dot`")

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
        )

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(self.in_channels),
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
            )

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "gaussian":
            self.theta = nn.Identity()

        # define theta and phi for all operations except gaussian
        if self.mode in ("embedded", "dot"):
            self.theta = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
            )
            padding_size = 1 if phi_ksize == 3 else 0
            self.phi = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=phi_ksize,
                padding=padding_size,
            )

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool3d(kernel_size=(1, 2, 2)))
            if hasattr(self, "phi"):
                self.phi = nn.Sequential(self.phi, nn.MaxPool3d(kernel_size=(1, 2, 2)))
            else:
                self.phi = nn.MaxPool3d(kernel_size=(1, 2, 2))
        else:
            if not hasattr(self, "phi"):
                self.phi = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args
            x: (B, C, T, H, W) for dimension=3; (B, C, H, W) for dimension 2; (B, C, T) for dimension 1
        """

        batch_size = x.shape[0]

        # (B, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = rearrange(self.g(x), "b c ... -> b (...) c")

        if self.mode in ("gaussian", "embedded", "dot"):
            theta_x = self.theta(x[:, :, -1])
            theta_x = rearrange(theta_x, "b c  ... -> b (...) c")
            phi_x = rearrange(self.phi(x), "b c ... -> b c (...)")
            f = theta_x @ phi_x

        if self.mode in ("gaussian", "embedded"):
            f_div_C = F.softmax(f, dim=-1)
        else:
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = rearrange(f_div_C @ g_x, "b f c -> b c f")
        y = y.view(batch_size, self.inter_channels, *x.size()[3:])

        return x[:, :, -1] + self.W_z(y)


class NonLocalBlock(TemporalCompression):
    def __init__(self, input_size: int, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        self.nlb = NLBlockND(
            self.input_size, mode="embedded", dimension=3, bn_layer=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nlb(x)[:, :, -1]

    @property
    def output_size(self) -> int:
        return self.input_size


class EfficientNonLocalBlock(TemporalCompression):
    def __init__(self, input_size: int, args: argparse.Namespace, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size
        phi_ksize = args.phi_ksize
        bn_layer = args.batch_norm_nlb
        sub_sample = args.sub_sample_nlb
        mode = args.nlb_mode
        self.nlb = EfficientNLBlockND(
            self.input_size,
            mode=mode,
            dimension=3,
            phi_ksize=phi_ksize,
            bn_layer=bn_layer,
            sub_sample=sub_sample,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nlb(x)

    @property
    def output_size(self) -> int:
        return self.input_size


if __name__ == "__main__":
    import itertools
    from utils.utils_scripts import Timer

    for bn_layer, sub_sample, mode in itertools.product(
        [True, False], [True, False], ["gaussian", "embedded", "dot"]
    ):
        img = torch.randn(2, 3, 8, 20, 20)
        net = EfficientNLBlockND(
            in_channels=3,
            mode=mode,
            dimension=3,
            bn_layer=bn_layer,
            sub_sample=sub_sample,
            phi_ksize=3,
        )
        with Timer(f"{bn_layer=} {sub_sample=} {mode=} phi_ksize=3", one_line=True):
            out = net(img)

        img = torch.randn(2, 3, 8, 20, 20)
        net = EfficientNLBlockND(
            in_channels=3,
            mode=mode,
            dimension=3,
            bn_layer=bn_layer,
            sub_sample=sub_sample,
            phi_ksize=1,
        )
        with Timer(f"{bn_layer=} {sub_sample=} {mode=} phi_ksize=1", one_line=True):
            out = net(img)
