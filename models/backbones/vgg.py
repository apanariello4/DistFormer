import argparse
import torchvision
import torch
from models.base_model import BaseModel


class VGG16(BaseModel):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.pretrained = args.pretrain
        self.use_centers = args.use_centers
        self.vgg = torchvision.models.vgg16(
            weights="DEFAULT" if self.pretrained else None
        )
        if self.pretrained:
            print("Using pretrained weights for VGG16")

        if self.use_centers:
            self.perform_surgery()
            self.MEAN += [0]
            self.STD += [1]

        self.frame_size = (23, 40)
        self.output_size = 512
        self.scale = 1 / 16

        del self.vgg.classifier
        del self.vgg.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = self.normalize_input(x)
        return self.vgg.features(x)

    def perform_surgery(self):
        new_conv = torch.nn.Conv2d(
            4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        new_conv.weight.data[:, :-1, :, :] = self.vgg.features[0].weight.data
        self.vgg.features[0] = new_conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    model = VGG16(args)
    print(model)

    x = torch.randn(2, 4, 6, 736, 1280).cuda()
    model = model.cuda()
    y = model(x)
    print(y.shape)
