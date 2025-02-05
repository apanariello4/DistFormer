import torch
import torch.nn as nn
import torchvision

from models.backbones.resnet_fpn import TopDownPathway


class I2D(nn.Module):
    def __init__(
        self, pretrained: bool = True, hdim: int = 256, hdim_red_factor: float = 0.5
    ) -> None:
        super().__init__()

        resnet = torchvision.models.resnet34(weights="DEFAULT" if pretrained else None)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        inplanes = [
            self.get_output_channels(layer)
            for layer in (self.layer1, self.layer2, self.layer3, self.layer4)
        ]

        self.topdownpath = TopDownPathway(
            inplanes, [None, 2, 4, 8], hdim, hdim_red_factor
        )

    def get_output_channels(self, resnet_block: nn.Module) -> int:
        return [
            x
            for x in list(list(resnet_block[-1].children())[-1].children())
            if isinstance(x, nn.Conv2d)
        ][-1].out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        vol = self.topdownpath(c2, c3, c4, c5)
        return vol


if __name__ == "__main__":
    B, C, W, H = 3, 3, 360, 640
    x = torch.rand((B, C, W, H))
    print(x.shape)
    net = I2D(pretrained=False)
    output = net(x)
    print(output.shape)
