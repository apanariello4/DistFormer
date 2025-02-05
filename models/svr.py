import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset.naive_features import SVRDataset
from dataset.nuscenes import SVRNuscenesDataset
from models.base_model import BaseLifter
from trainers.base_trainer import Trainer
from trainers.trainer_mlp import TrainerMLP


class SVR(BaseLifter):
    TEMPORAL = False

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.distance_estimator = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.distance_estimator(x).squeeze()

    def get_loss_fun(self) -> nn.Module:
        return nn.L1Loss()

    def get_trainer(self) -> Trainer:
        return TrainerMLP

    def get_dataset(self, args) -> Dataset:
        return {
            "motsynth": SVRDataset,
            "nuscenes": SVRNuscenesDataset,
        }[args.dataset]

    def get_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.args.lr)
