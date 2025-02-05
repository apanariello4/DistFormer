from typing import Optional

import torch
import torch.nn as nn
import torchvision
from .regressor_utils import flatten_bboxes, get_last_frame, split_by_frame
from einops import rearrange, reduce

from models.bb_regressors.self_attention import SelfAttention


class ROITransformerV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        pool_size: int = 4,
        dropout: float = 0.0,
        transformer_aggregate: bool = True,
        roi_op: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.roi_op = roi_op if roi_op is not None else torchvision.ops.roi_pool
        self.pool_window = (
            self.pool_size**2
            if isinstance(self.pool_size, int)
            else self.pool_size[0] * self.pool_size[1]
        )
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_heads = 4
        self.hidden_dim = self.output_dim * 2

        self.aggregate_patches = transformer_aggregate
        embed = None if transformer_aggregate else "positional"

        self.self_attention = SelfAttention(
            num_heads=self.num_heads,
            norm=True,
            embed_dim=self.hidden_dim,
            embed=embed,
            init=False,
            dropout=dropout,
        )

        self.standard_projector = nn.Sequential(
            nn.Conv2d(
                self.input_dim,
                min(256, self.hidden_dim),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ELU(inplace=True),
            nn.Conv2d(
                min(256, self.hidden_dim),
                self.hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(
        self,
        x,
        bboxes: list,
        scale: Optional[float] = None,
        return_features: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """input shape: (B, C, T, H, W),
        bboxes shape: (B, N, 4)"""
        B, C, T, H, W = x.shape

        x = rearrange(x, "b c t h w -> (b t) c h w")
        flattened_bboxes = flatten_bboxes(bboxes)
        n_omini_per_frame = [len(b) for b in flattened_bboxes]

        roi_feat = self.get_roi_features(x, flattened_bboxes, B * T, scale)

        omini = split_by_frame(roi_feat, n_omini_per_frame)

        omini_batches, lens = get_last_frame(omini, B, T)

        x = torch.cat(omini_batches, dim=0)

        x = self.standard_projector(x)

        if not self.aggregate_patches:
            x = rearrange(x, "(b t) c h w -> (b t h w) c", t=T)
            lens = [len * self.pool_window for len in lens]
        else:
            x = reduce(x, "(b t) c h w -> (b t) c", "mean", t=T)

        x = list(torch.split(x, lens, dim=0))

        x = [x_ for x_ in x if len(x_) > 0]

        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)

        padding_mask, attn_mask = self.get_masks(
            max_len=x.shape[1], batch_size=B, lens=lens, device=x.device
        )

        last_frame_bbox = [b[-1] for b in bboxes if len(b) > 0]
        x = self.self_attention(
            x, key_padding_mask=padding_mask, bboxes=last_frame_bbox, H=H, W=W
        )
        x = x[padding_mask == False]

        if not self.aggregate_patches:
            totl = 0
            xnew = torch.zeros((sum(lens), self.hidden_dim), device=x.device)
            for len in lens:
                xnew[totl : totl + len] = x[totl : totl + len]
                totl += len
            x = reduce(
                xnew,
                "(b t h w) c -> (b t) c",
                "mean",
                t=T,
                h=self.pool_size[0],
                w=self.pool_size[1],
            )
            # xnew = self.projector(xnew)

        out = self.output(x)

        if torch.isnan(out).any():
            pass

        return out if not return_features else (out, roi_feat)

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

    def get_masks(
        self, max_len: int, batch_size: int, lens: list, device: torch.device
    ):
        padding_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=device
        )

        for b, len in enumerate(lens):
            padding_mask[b, len:] = True

        good = torch.tensor(lens) > 0
        return padding_mask[good], None
