import torch
import torch.nn as nn

from utils.graphs import get_adjacency_threshold_coords


class PREG(nn.Module):
    def __init__(self, threshold: float, **kwargs) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self, x: torch.Tensor, head_coords: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        curr_len = 0
        losses = torch.zeros(len(head_coords), device=x.device)

        for i, coords in enumerate(head_coords):
            if len(coords) == 0:
                continue
            curr_ytrue = y_true[curr_len : curr_len + len(coords)]
            curr_x = x[curr_len : curr_len + len(coords)]
            curr_len += len(coords)
            with torch.no_grad():
                A = get_adjacency_threshold_coords(
                    curr_ytrue, coords, threshold=self.threshold
                )
                D = torch.diag(torch.sum(A, dim=1))
                A = torch.inverse(D) @ A

            losses[i] = torch.mean((A @ curr_x - curr_x) ** 2) / len(coords)
        return losses.sum()
