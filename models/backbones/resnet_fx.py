# -*- coding: utf-8 -*-
# ---------------------


import torch
from models.backbones.resnet_models import resnet18, resnet50, resnet101, resnet152

from models.base_model import BaseLifter


DEPTH_ACCEPTED_VALUES = {"18", "50", "101", "152"}

# RGB mean and std of the ImageNet dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetFX(BaseLifter):
    TEMPORAL = False

    def __init__(self, resnet_depth="resnet18", pretrained=True):
        """
        ResNet with just the feature extraction blocks.
            >> input: image tensor with shape (B, C, H, W)
            >> output: features with shape (B, 512, H/32, W/32)
        :param resnet_depth: values in {'18', '50', '101', '152'}
        :param pretrained: do you want use pretrained model on ImageNet?
            >> NOTE: if pretrained is `True`, input RGB images shoud be in range [0, 1]
               and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
               You can use the following transform to normalize:
               norm_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        """
        super().__init__()

        self.pretrained = pretrained

        resnet_depth = str(resnet_depth)
        if resnet_depth == "18":
            self.resnet = resnet18(pretrained=pretrained)
        elif resnet_depth == "50":
            self.resnet = resnet50(pretrained=pretrained)
        elif resnet_depth == "101":
            self.resnet = resnet101(pretrained=pretrained)
        elif resnet_depth == "152":
            self.resnet = resnet152(pretrained=pretrained)
        else:
            raise ValueError(
                f"wrong value for `resnet_depth`; "
                f"it must be one of {DEPTH_ACCEPTED_VALUES}, not '{resnet_depth}'"
            )

    @staticmethod
    def normalize_input(x):
        x = x.clone()
        for i in range(3):
            x[:, i] = (x[:, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
        return x

    def forward(self, x):
        if x.dim() == 5:
            assert x.shape[2] == 1, "tried to remove time dimension but it was not 1"
            x = x.squeeze(2)

        if self.pretrained:
            x = self.normalize_input(x)

        # first block
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # other blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x


# ---------


def main():
    import time
    import torchsummary

    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetFX(resnet_depth=18).to(device)

    x = torch.rand((batch_size, 3, 1280, 736)).to(device)
    torchsummary.summary(model=model, input_size=x.shape[1:], device=str(device))

    t = time.time()
    y = model.forward(x)
    t = time.time() - t

    print(f"$> input shape: {tuple(x.shape)}")
    print(f"$> output shape: {tuple(y.shape)}")
    print(f"$> forward time: {t:.4f} s with a batch size of {batch_size}")
    print(f"$> scale factor (H, W): {x.shape[2] / y.shape[2], x.shape[3] / y.shape[3]}")


if __name__ == "__main__":
    main()
