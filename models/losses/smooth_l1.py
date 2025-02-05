import torch.nn as nn
import torch


class SmoothL1(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.smooth = nn.SmoothL1Loss(beta=beta)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.smooth(y_pred, y_true)
