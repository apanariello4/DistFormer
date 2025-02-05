from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn


class ROIPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2048,
        pool_size: int = 4,
        detector: bool = True,
        roi_op=None,
        **kwargs,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.roi_op = roi_op
        self.pool_window = (
            self.pool_size**2
            if isinstance(self.pool_size, int)
            else self.pool_size[0] * self.pool_size[1]
        )

        if detector:
            self.detector = nn.Sequential(
                nn.Conv2d(input_dim, 256, kernel_size=1, stride=1, padding=0),
                nn.ELU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                nn.ELU(inplace=True),
                nn.Flatten(),
                nn.Linear(128 * self.pool_window, output_dim),
                nn.ELU(inplace=True),
            )

    def forward(
        self, x, bboxes: list, scale: Optional[None] = None, **kwargs
    ) -> torch.Tensor:
        """input shape: (B, C, T, H, W),
        bboxes shape: (B, N, 4)"""
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.get_roi_features(x, bboxes, B * T, scale)

        if hasattr(self, "detector"):
            x = self.detector(x)
        return x

    def get_roi_features(
        self, x, bboxes: list, n_frames: int, scale: float
    ) -> torch.Tensor:
        roi_indices = torch.repeat_interleave(
            torch.arange(0, n_frames),
            torch.tensor([len(b) for b in bboxes], requires_grad=False),
        ).to(x.device)
        roi_bboxes = torch.cat(bboxes).to(x.device)
        rois = torch.cat([roi_indices[:, None].float(), roi_bboxes], dim=1)

        return self.roi_op(
            input=x,
            boxes=rois.type_as(x),
            output_size=self.pool_size,
            spatial_scale=scale,
        )
