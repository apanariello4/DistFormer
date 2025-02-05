import itertools
import torch
import torch.nn as nn
import wandb

from utils.utils_scripts import get_center
from utils.utils_scripts import from2dot5dto3d

FOCAL = 1158
C_X = 960
C_Y = 540


class EnhancedZHU(nn.Module):
    def __init__(self, alpha: float = 0.05, train: bool = True):
        super().__init__()
        self.smooth = nn.SmoothL1Loss()
        self.alpha = alpha
        self.training = train

    def forward(
        self,
        y_pred: tuple[torch.Tensor, torch.Tensor],
        y_true: torch.Tensor,
        bboxes: tuple[torch.Tensor],
    ) -> torch.Tensor:
        if self.training:
            z, k = y_pred
            loss = self.smooth(z, y_true)
            centers = [[get_center(bb) for bb in batch] for batch in bboxes]
            centers = torch.tensor(
                list(itertools.chain.from_iterable(centers)), device=y_true.device
            )

            k = k + torch.ones_like(k)
            k = k * (
                torch.ones_like(k)
                * torch.tensor([C_X, C_Y]).view(1, 2).to(y_true.device)
            )

            _3D_points_pred = from2dot5dto3d(
                torch.cat([k, z.unsqueeze(1)], dim=1), f=(FOCAL, FOCAL), c=(C_X, C_Y)
            )
            _3D_points_true = from2dot5dto3d(
                torch.cat([centers, y_true.unsqueeze(1)], dim=1),
                f=(FOCAL, FOCAL),
                c=(C_X, C_Y),
            )
            regressor_loss = self.alpha * (
                torch.mean(
                    torch.linalg.norm(_3D_points_pred - _3D_points_true, dim=1) / y_true
                )
                / len(y_true)
            )
            if wandb.run:
                wandb.log({"zhu_enhanced_regressor_loss": regressor_loss.item()})
            return loss + regressor_loss

        return self.smooth(y_pred, y_true)
