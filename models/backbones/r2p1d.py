from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from models.base_model import BaseModel


def get_inplanes():
    return [64, 128, 256, 512]


def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        mid_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False,
    )


def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(
        mid_planes,
        planes,
        kernel_size=(3, 1, 1),
        stride=(stride, 1, 1),
        padding=(1, 0, 0),
        bias=False,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
        frozen_stages=-1,
        pretrained=False,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.pretrained = pretrained

        assert frozen_stages in (-1, 0, 1, 2, 3, 4)
        self.frozen_stages = frozen_stages

        self.layer0 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1_s",
                        nn.Conv3d(
                            n_input_channels,
                            mid_planes,
                            kernel_size=(1, 7, 7),
                            stride=(1, 2, 2),
                            padding=(0, 3, 3),
                            bias=False,
                        ),
                    ),
                    ("bn1_s", nn.BatchNorm3d(mid_planes)),
                    ("relu1_s", nn.ReLU(inplace=True)),
                    (
                        "conv1_t",
                        nn.Conv3d(
                            mid_planes,
                            self.in_planes,
                            kernel_size=(conv1_t_size, 1, 1),
                            stride=(conv1_t_stride, 1, 1),
                            padding=(conv1_t_size // 2, 0, 0),
                            bias=False,
                        ),
                    ),
                    ("bn1_t", nn.BatchNorm3d(self.in_planes)),
                    ("relu1_t", nn.ReLU(inplace=True)),
                    (
                        "maxpool1",
                        nn.Sequential()
                        if no_max_pool
                        else nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                    ),
                ]
            )
        )

        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.num_features = block_inplanes[3] * block.expansion
        self.fc = nn.Linear(self.num_features, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def load_state_dict_from_repo_version(self, checkpoint):
        edited_dict = checkpoint["state_dict"]
        remaped_key = {
            s: "layer0." + s
            for s in list(edited_dict.keys())
            if not s.startswith("layer") and not s.startswith("fc")
        }
        state_dict = dict(
            (remaped_key[key], edited_dict[key]) if key in remaped_key else (key, value)
            for key, value in edited_dict.items()
        )

        self.load_state_dict(state_dict)

    # def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
    #     downsample = None
    #     if stride != 1 or self.in_planes != planes * block.expansion:
    #         if shortcut_type == 'A':
    #             downsample = partial(self._downsample_basic_block,
    #                                  planes=planes * block.expansion,
    #                                  stride=stride)
    #         else:
    #             downsample = nn.Sequential(
    #                 conv1x1x1(self.in_planes, planes * block.expansion, stride),
    #                 nn.BatchNorm3d(planes * block.expansion))

    #     layers = []
    #     layers.append(
    #         block(self.in_planes, planes, stride, downsample, shortcut_type))
    #     self.in_planes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.in_planes, planes))

    #     return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        for i in range(5):
            layer = getattr(self, "layer{}".format(i))
            if i <= self.frozen_stages:
                with torch.no_grad():
                    x = layer(x)
            else:
                x = layer(x)

        return x

    def get_last_layer(self):
        return self.layer4

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(0, self.frozen_stages + 1):
                m = getattr(self, "layer{}".format(i))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


def generate_model(model_depth, **kwargs):
    assert model_depth in (10, 18, 34, 50, 101, 152, 200)
    n_input_channels = kwargs.get("n_input_channels", None)

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        pretrained = kwargs.get("pretrained")
        pretrained_w = (
            torchvision.models.video.R2Plus1D_18_Weights if pretrained else None
        )
        model = torchvision.models.video.r2plus1d_18(weights=pretrained_w)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return ResNetWrapper(model, n_input_channels) if model_depth == 18 else model


class ResNetWrapper(nn.Module):
    def __init__(self, model, n_input_channels):
        super().__init__()
        self.model = model
        self.n_input_channels = n_input_channels
        self.resize = torchvision.transforms.Resize(224)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2),
            nn.Upsample((368, 640), mode="bilinear", align_corners=True),
        )

        # update the first layer keeping the weights
        if n_input_channels != 3:
            conv1 = nn.Conv3d(
                n_input_channels,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            )
            # copy weights from pretrained model
            conv1.weight.data[:, :3, :, :, :] = self.model.stem[0].weight.clone()
            self.model.stem[0] = conv1

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = torch.stack([self.resize(y) for y in x])
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.up(x.squeeze(2))

        return x


if __name__ == "__main__":
    pass
