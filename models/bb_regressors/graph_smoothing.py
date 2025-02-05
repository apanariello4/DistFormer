import torch
import torch.nn as nn

from utils.graphs import get_adjacency_threshold_coords, get_laplace_matrix


class GraphSmoothing(nn.Module):
    def __init__(
        self,
        threshold: float = 1.0,
        normalize_laplacian: bool = True,
        sample: bool = False,
    ):
        super().__init__()
        self.threshold = threshold
        self.normalize_laplacian = normalize_laplacian
        self.sample = sample

    def forward(
        self, x: torch.Tensor, head_coords: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if not self.sample:
            x = x[0] if isinstance(x, tuple) else x
            # x = x.unsqueeze(1)
        elif self.sample:
            assert isinstance(x, tuple), "x must be a tuple of mu and logvar"
            mu, logvar = x
            x = mu + torch.exp(logvar * 0.5) * torch.randn_like(mu)

        curr_len = 0
        losses = []

        for i, coords in enumerate(head_coords):
            if len(coords) == 0:
                continue
            curr_ytrue = y_true[curr_len : curr_len + len(coords)]
            curr_x = x[curr_len : curr_len + len(coords)]
            A = get_adjacency_threshold_coords(
                curr_ytrue, coords, threshold=self.threshold
            )
            L = get_laplace_matrix(A, normalize=self.normalize_laplacian)
            losses.append((curr_x.view(1, -1) @ L @ curr_x).squeeze())
            curr_len += len(coords)

        return torch.stack(losses).sum()
