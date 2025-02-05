import torch
import torch.nn as nn


class GNLL(nn.Module):
    def __init__(self, use_constant: bool = False):
        super().__init__()
        self.EPS = torch.finfo(torch.float32).eps
        self.gnll = nn.GaussianNLLLoss(reduction="sum", full=use_constant)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        mean, logvar = y_pred
        assert mean.shape == logvar.shape == y_true.shape

        return self.gnll(input=mean, var=logvar.exp(), target=y_true) / (
            len(y_true) + self.EPS
        )
