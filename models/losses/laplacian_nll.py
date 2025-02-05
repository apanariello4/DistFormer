from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LNLL(nn.Module):
    def __init__(self, use_constant: bool = False, reduction: Optional[str] = "mean"):
        super().__init__()
        self.EPS = torch.finfo(torch.float32).eps
        self.constant = np.log(2) if use_constant else 0
        self.reduction = {"mean": torch.mean, "sum": torch.sum, None: lambda x: x}[
            reduction
        ]

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        mean, logvar = y_pred
        assert mean.shape == logvar.shape == y_true.shape

        a = torch.abs(1 - mean / y_true) * torch.exp(-logvar) + self.EPS
        if not self.training:
            return torch.abs(1 - mean / y_true).mean(), torch.exp(logvar).mean()
        loss = a + logvar + self.constant

        return self.reduction(loss)


if __name__ == "__main__":
    loss = LNLL(True, "sum")
    y_pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    y_true = torch.tensor([1.0, 2.0, 3.0])
    loss = loss(y_pred, y_true)
    print(loss)
