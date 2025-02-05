import argparse

import torch
import torchvision
from torch.hub import load_state_dict_from_url

from models.base_model import BaseModel

MODELS = {
    "tiny": torchvision.models.convnext.convnext_tiny,
    "small": torchvision.models.convnext.convnext_small,
    "base": torchvision.models.convnext.convnext_base,
    "large": torchvision.models.convnext.convnext_large,
}

MODELS_22K_PATH = {
    "tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
}

OUT_CHANNELS = {"tiny": 768, "small": 768, "base": 1024, "large": 1536}


class ConvNeXt(BaseModel):
    def __init__(self, args: argparse.Namespace, size: str):
        super().__init__()
        self.pretrained = args.pretrain
        self.use_centers = args.use_centers
        self.residual_centers = args.residual_centers
        if self.residual_centers:
            print(
                f"Residual centers are not supported for {self.__class__.__name__}. Turning off."
            )

        self.convnext = MODELS[size](
            weights="DEFAULT" if self.pretrained == "imagenet" else None
        )
        del self.convnext.classifier
        del self.convnext.avgpool

        if self.pretrained:
            if self.pretrained == "imagenet22k":
                # weights for imagenet22k are not available in torchvision => load them manually
                state_dict = load_state_dict_from_url(
                    MODELS_22K_PATH[size], progress=True
                )
                self.weights_surgery(state_dict["model"], size=size)
                self.convnext.load_state_dict(state_dict["model"])
                print(f"Using IMAGENET22K weights for ConvNeXt_{size}")
            else:
                print(f"Using IMAGENET1K weights for ConvNeXt_{size}")

        if self.use_centers:
            self.perform_surgery()
            self.MEAN += [0]
            self.STD += [1]

        self.output_size = OUT_CHANNELS[size]
        self.scale = 1 / 32
        self.frame_size = self.get_frame_size(args)

    def get_frame_size(self, args: argparse.Namespace) -> tuple:
        h = args.input_h_w[0] * self.scale
        w = args.input_h_w[1] * self.scale
        return int(h), int(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = self.normalize_input(x)
        return self.convnext.features(x)

    def perform_surgery(self):
        t = self.convnext.features[0][0]
        new_conv = torch.nn.Conv2d(4, t.out_channels, t.kernel_size, t.stride)
        new_conv.weight.data[:, :-1, :, :] = self.convnext.features[0][0].weight.data
        self.convnext.features[0][0] = new_conv

    def weights_surgery(self, state_dict, size):
        """
        This function is used to convert the weights from the original ConvNeXt implementation to the torchvision implementation.
        """
        block_indexes = {1: 0, 3: 1, 5: 2, 7: 3}
        downsample_indexes = {0: 0, 2: 1, 4: 2, 6: 3}

        def replace_downsample_layers(state_dict, index):
            state_dict[f"features.{index}.0.weight"] = state_dict[
                f"downsample_layers.{downsample_indexes[index]}.0.weight"
            ]
            state_dict[f"features.{index}.0.bias"] = state_dict[
                f"downsample_layers.{downsample_indexes[index]}.0.bias"
            ]
            state_dict[f"features.{index}.1.weight"] = state_dict[
                f"downsample_layers.{downsample_indexes[index]}.1.weight"
            ]
            state_dict[f"features.{index}.1.bias"] = state_dict[
                f"downsample_layers.{downsample_indexes[index]}.1.bias"
            ]
            del state_dict[f"downsample_layers.{downsample_indexes[index]}.0.weight"]
            del state_dict[f"downsample_layers.{downsample_indexes[index]}.0.bias"]
            del state_dict[f"downsample_layers.{downsample_indexes[index]}.1.weight"]
            del state_dict[f"downsample_layers.{downsample_indexes[index]}.1.bias"]

        def replace_gamma(state_dict, index, repeat):
            for i in range(repeat):
                state_dict[f"features.{index}.{i}.layer_scale"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.gamma"
                ]
                del state_dict[f"stages.{block_indexes[index]}.{i}.gamma"]

        def replace_block(state_dict, index, repeat):
            for i in range(repeat):
                state_dict[f"features.{index}.{i}.block.0.weight"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.dwconv.weight"
                ]
                state_dict[f"features.{index}.{i}.block.0.bias"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.dwconv.bias"
                ]
                state_dict[f"features.{index}.{i}.block.2.weight"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.norm.weight"
                ]
                state_dict[f"features.{index}.{i}.block.2.bias"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.norm.bias"
                ]
                state_dict[f"features.{index}.{i}.block.3.weight"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.pwconv1.weight"
                ]
                state_dict[f"features.{index}.{i}.block.3.bias"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.pwconv1.bias"
                ]
                state_dict[f"features.{index}.{i}.block.5.weight"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.pwconv2.weight"
                ]
                state_dict[f"features.{index}.{i}.block.5.bias"] = state_dict[
                    f"stages.{block_indexes[index]}.{i}.pwconv2.bias"
                ]
                del state_dict[f"stages.{block_indexes[index]}.{i}.dwconv.weight"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.dwconv.bias"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.norm.weight"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.norm.bias"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.pwconv1.weight"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.pwconv1.bias"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.pwconv2.weight"]
                del state_dict[f"stages.{block_indexes[index]}.{i}.pwconv2.bias"]

        replace_downsample_layers(state_dict, 0)
        replace_downsample_layers(state_dict, 2)
        replace_downsample_layers(state_dict, 4)
        replace_downsample_layers(state_dict, 6)

        # search all keys that has layer_scale inside and change shape from [K] to [K,1,1]
        for key in list(state_dict.keys()):
            if "gamma" in key:
                state_dict[key] = state_dict[key].unsqueeze(1).unsqueeze(1)

        replace_gamma(state_dict, 1, 3)
        replace_gamma(state_dict, 3, 3)
        replace_gamma(state_dict, 5, 9 if size == "tiny" else 27)
        replace_gamma(state_dict, 7, 3)

        replace_block(state_dict, 1, 3)
        replace_block(state_dict, 3, 3)
        replace_block(state_dict, 5, 9 if size == "tiny" else 27)
        replace_block(state_dict, 7, 3)

        del state_dict["norm.weight"]
        del state_dict["norm.bias"]
        del state_dict["head.weight"]
        del state_dict["head.bias"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="imagenet22k")
    parser.add_argument("--use_centers", type=int, default=1)
    parser.add_argument("--residual_centers", type=int, default=0)
    parser.add_argument("--input_h_w", type=int, nargs="+", default=[720, 1280])
    args = parser.parse_args()

    model = ConvNeXt(args, "tiny")
    # print(model)

    x = torch.randn(2, 3 if args.use_centers == 0 else 4, 720, 1280).cuda()
    model = model.cuda()
    y = model(x)
    print(y.shape)
