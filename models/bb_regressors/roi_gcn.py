import torch.nn as nn
import torchvision
import torch

from models.gcn import GCN
from utils.graphs import get_adjacency_dist, get_adjacency_iou


adjacency_funcs = {"iou": get_adjacency_iou, "dist": get_adjacency_dist}


class ROIGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        adjacency: str = "iou",
        pool_size: int = 4,
        scale: float = 1 / 16,
        **kwargs,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.adjacency = adjacency_funcs[adjacency]
        self.roi_pooling_op = torchvision.ops.RoIPool(self.pool_size, scale)

        self.detector = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * self.pool_size**2, output_dim),
            nn.ELU(inplace=True),
        )

        self.graph = GCN(
            input_dim=output_dim,
        )

    def forward(self, x, bboxes: list, **kwargs) -> torch.Tensor:
        """input shape: (B, C, T, H, W),
        bboxes shape: (B, N, 4)"""
        B = x.shape[0]
        roi_indices = torch.repeat_interleave(
            torch.arange(0, B), torch.tensor([len(b) for b in bboxes])
        )
        roi_bboxes = torch.cat(bboxes)
        rois = torch.cat([roi_indices[:, None].float(), roi_bboxes], dim=1).to(x.device)
        roi_feature_maps = self.roi_pooling_op(x, rois)

        x = self.detector(roi_feature_maps)
        adj = self.adjacency(bboxes, x.device)

        x = self.graph(x, adj)

        return x
