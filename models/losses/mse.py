import torch
import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = torch.finfo(torch.float32).eps

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        assert len(y_pred) == len(y_true)
        assert y_pred.shape == y_true.shape

        squared_sum = ((y_true - y_pred) ** 2).sum()

        return squared_sum / (len(y_true) + self.EPS)
