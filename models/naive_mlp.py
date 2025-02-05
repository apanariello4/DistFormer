import torch.nn as nn
import torch

from trainers.base_trainer import Trainer
from dataset.naive_features import MLPDataset
from models.base_model import BaseLifter
from torch.utils.data import Dataset
from trainers.trainer_mlp import TrainerMLP


class MLP(BaseLifter):
    TEMPORAL = False

    def __init__(
        self, args, input_dim: int = 5, hidden_dim: int = 256, output_dim: int = 1
    ):
        super(MLP, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def get_loss_fun(self) -> nn.Module:
        return nn.L1Loss()

    def get_trainer(self) -> Trainer:
        return TrainerMLP

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def get_dataset(self, args) -> Dataset:
        return {"motsynth": MLPDataset}[args.dataset]
