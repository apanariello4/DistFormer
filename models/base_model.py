import argparse
import math

import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize

from trainers.base_trainer import Trainer


class BaseModel(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()

    def kaiming_init(self, activation: str):
        """
        Apply "Kaiming-Normal" initialization to all Conv2D(s) of the model.
        :param activation: activation function after conv; values in {'relu', 'leaky_relu'}
        :return:
        """
        assert activation in {
            "ReLU",
            "LeakyReLU",
            "Swish",
            "leaky_relu",
        }, "`activation` must be 'ReLU', 'LeakyReLU'"

        if activation == "LeakyReLU":
            activation = "leaky_relu"
        activation = activation.lower()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=activation
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch of RGB images with values in [0, 1]
        :return: normalized batch using ImageNet mean and std
        """
        return normalize(x, mean=self.MEAN, std=self.STD)

    @property
    def n_param(self) -> int:
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> str:
        """
        :return: string that represents the device on which the model is currently located
            >> e.g.: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
        """
        return str(next(self.parameters()).device)

    @property
    def is_cuda(self) -> bool:
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return "cuda" in self.current_device

    def get_frame_size(self, args: argparse.Namespace) -> tuple:
        h = args.input_h_w[0] * self.scale
        w = args.input_h_w[1] * self.scale

        h = math.ceil(h)
        w = math.ceil(w)

        return h, w

    @staticmethod
    def get_params_from_conv(target_conv: nn.Conv2d):
        """return in_ch, out_ch, stride, padding, kernel_size

        Args:
            myconv (nn.Conv2d): target conv

        Returns:
            tuple: in_ch, out_ch, stride, padding, kernel_size
        """
        return (
            target_conv.weight.shape[1],
            target_conv.weight.shape[0],
            target_conv.stride,
            target_conv.padding,
            tuple(target_conv.weight.shape[2:]),
        )


class BaseLifter(BaseModel):
    TEMPORAL: bool = None

    def __init__(self):
        super().__init__()

        if self.TEMPORAL is None:
            raise NotImplementedError("TEMPORAL must be set to True or False")

    def get_loss_fun(self) -> nn.Module:
        raise NotImplementedError

    def get_trainer(self) -> Trainer:
        raise NotImplementedError

    def get_optimizer(self, args: argparse.Namespace) -> torch.optim.Optimizer:
        raise NotImplementedError

    def load_w(self, path: str, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location="cpu")

        self.load_state_dict(state_dict["weights"], strict=strict)

        self.ds_stats = state_dict["ds_stats"]

    def save_w(self, path: str, cnf) -> None:
        state_dict = {
            "weights": self.state_dict(),
            "ds_stats": self.ds_stats,
            "cnf": cnf,
        }
        torch.save(state_dict, path)
