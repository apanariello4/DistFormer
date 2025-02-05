import torch
import torch.nn as nn


class MaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.EPS = torch.finfo(torch.float32).eps

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, binary_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Masked mean squared error loss.
        """
        squared_errors = (y_true - y_pred) ** 2
        masked_squared_errors = squared_errors * binary_mask
        masked_mse = masked_squared_errors.sum() / (binary_mask.sum() + self.EPS)
        return masked_mse
