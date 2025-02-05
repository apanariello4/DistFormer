from functools import partial

from models.backbones.convnext import ConvNeXt
from models.backbones.convnext_fpn import ConvNeXtFPN
from models.backbones.convnext_v2 import CustomConvNeXtV2
from models.backbones.convnext_v2_fpn import ConvNeXtV2FPN
from models.backbones.efficientnet import EfficientNet
from models.backbones.mobilenet import MobileNet
from models.backbones.resnet import ResNet
from models.backbones.resnet_bam import ResNetBAM
from models.backbones.resnet_fpn import ResNetFPN
from models.backbones.shufflenet import ShuffleNet
from models.backbones.vgg import VGG16
from models.backbones.vit import ViT

BACKBONES = {
    "resnet18": partial(ResNet, depth=18),
    "resnet34": partial(ResNet, depth=34),
    "resnet50": partial(ResNet, depth=50),
    "resnet101": partial(ResNet, depth=101),
    "resnet152": partial(ResNet, depth=152),
    "resnetfpn18": partial(ResNetFPN, depth=18),
    "resnetfpn34": partial(ResNetFPN, depth=34),
    "resnetfpn50": partial(ResNetFPN, depth=50),
    "resnetfpn101": partial(ResNetFPN, depth=101),
    "resnetfpn152": partial(ResNetFPN, depth=152),
    "resnetbam18": partial(ResNetBAM, depth=18),
    "resnetbam34": partial(ResNetBAM, depth=34),
    "resnetbam50": partial(ResNetBAM, depth=50),
    "resnetbam101": partial(ResNetBAM, depth=101),
    "resnetbam152": partial(ResNetBAM, depth=152),
    "vgg16": VGG16,
    "efficientnet-b0": EfficientNet,
    "efficientnet-b1": EfficientNet,
    "efficientnet-b2": EfficientNet,
    "efficientnet-b3": EfficientNet,
    "efficientnet-b4": EfficientNet,
    "efficientnet-b5": EfficientNet,
    "efficientnet-b6": EfficientNet,
    "efficientnet-b7": EfficientNet,
    "efficientnet_v2_s": EfficientNet,
    "efficientnet_v2_m": EfficientNet,
    "efficientnet_v2_l": EfficientNet,
    "shufflenet_v2_x0_5": ShuffleNet,
    "shufflenet_v2_x1_0": ShuffleNet,
    "shufflenet_v2_x1_5": ShuffleNet,
    "shufflenet_v2_x2_0": ShuffleNet,
    "mobilenet_v2": MobileNet,
    "mobilenet_v3_large": MobileNet,
    "mobilenet_v3_small": MobileNet,
    "vit": ViT,
    "convnext_tiny": partial(ConvNeXt, size="tiny"),
    "convnext_small": partial(ConvNeXt, size="small"),
    "convnext_base": partial(ConvNeXt, size="base"),
    "convnext_large": partial(ConvNeXt, size="large"),
    "convnextfpn_tiny": partial(ConvNeXtFPN, size="tiny"),
    "convnextfpn_small": partial(ConvNeXtFPN, size="small"),
    "convnextfpn_base": partial(ConvNeXtFPN, size="base"),
    "convnextfpn_large": partial(ConvNeXtFPN, size="large"),
    "convnextv2_tiny": partial(CustomConvNeXtV2, size="tiny"),
    "convnextv2_base": partial(CustomConvNeXtV2, size="base"),
    "convnextv2_large": partial(CustomConvNeXtV2, size="large"),
    "convnextv2fpn_tiny": partial(ConvNeXtV2FPN, size="tiny"),
    "convnextv2fpn_base": partial(ConvNeXtV2FPN, size="base"),
    "convnextv2fpn_large": partial(ConvNeXtV2FPN, size="large"),
}
