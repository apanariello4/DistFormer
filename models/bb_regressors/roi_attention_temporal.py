from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from einops import rearrange


class ROITimeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        pool_size: int = 4,
        scale: float = 1 / 16,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.roi_pooling_op = torchvision.ops.RoIPool(self.pool_size, scale)
        self.pool_window = (
            self.pool_size**2
            if isinstance(self.pool_size, int)
            else self.pool_size[0] * self.pool_size[1]
        )

        self.detector = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1, stride=1, padding=0),
        )

        self.past_detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.pool_window, 128),
            nn.ELU(inplace=True),
        )

        self.attn = torch.nn.TransformerDecoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, batch_first=True, dropout=dropout
        )

        self.projector = nn.Sequential(
            nn.Conv2d(
                128, output_dim // self.pool_window, kernel_size=1, stride=1, padding=0
            ),
            nn.Flatten(),
        )

    def forward(
        self, x, bboxes: list, scale: Optional[None] = None, **kwargs
    ) -> torch.Tensor:
        """input shape: (B, C, T, H, W),
        bboxes shape: (B, N, 4)"""
        B, C, T, H, W = x.shape

        x = rearrange(x, "b c t h w -> (b t) c h w")
        flattened_bboxes = chain.from_iterable(bboxes)

        for i in range(len(flattened_bboxes)):
            if flattened_bboxes[i].ndim > 2:
                flattened_bboxes[i] = flattened_bboxes[i].squeeze(0)

        omini_per_frame = [len(b) for b in flattened_bboxes]

        roi_indices = torch.repeat_interleave(
            torch.arange(0, B * T),
            torch.tensor([len(b) for b in flattened_bboxes], requires_grad=False),
        )
        roi_bboxes = torch.cat(flattened_bboxes)
        rois = torch.cat([roi_indices[:, None].float(), roi_bboxes], dim=1).to(x.device)

        if scale is None:
            x = self.roi_pooling_op(x, rois.type_as(x))
        else:
            x = torchvision.ops.RoIPool(self.pool_size, scale)(x, rois.type_as(x))

        if hasattr(self, "detector"):
            x = self.detector(x)

        curr_omini = 0
        omini = []
        for frame in omini_per_frame:
            omini.append(x[curr_omini : curr_omini + frame])
            curr_omini += frame

        last_frame_omini = []
        past_frames_omini = []

        lens_last_frame = []
        lens_past_frames = []

        for b in range(B):
            curr = omini[b * T : (b + 1) * T]
            past_frames = curr
            last_frame = curr[-1]

            last_frame = rearrange(last_frame, "n f h w -> (n h w) f")

            last_frame_omini.append(last_frame)
            past_frames_omini.append(past_frames)

            lens_last_frame.append(len(last_frame))
            lens_past_frames.append([len(o) for o in past_frames])

        memory = chain.from_iterable(past_frames_omini)
        memory = torch.nested.nested_tensor(
            memory, dtype=torch.float32, requires_grad=True
        )
        memory = torch.nested.to_padded_tensor(memory, padding=0.0)
        memory = rearrange(memory, "b n f h w -> (b n) f h w")
        memory = self.past_detector(memory)
        memory = rearrange(memory, "(b n) f -> b n f", b=B)

        x = torch.nested.nested_tensor(
            last_frame_omini, dtype=torch.float32, requires_grad=True
        )
        x = torch.nested.to_padded_tensor(x, padding=0.0)

        Lmax = x.shape[1]
        Lmemmax = memory.shape[1]

        attn_mask_q = torch.zeros((B, Lmax), dtype=torch.bool, device=x.device)
        attn_mask_mem = torch.zeros((B, Lmemmax), dtype=torch.bool, device=x.device)

        max_pad = Lmemmax // T

        for b, l_lf, l_pfs in zip(range(B), lens_last_frame, lens_past_frames):
            attn_mask_q[b, l_lf:] = True
            for t in range(T):
                attn_mask_mem[b, t * max_pad + l_pfs[t] : (t + 1) * max_pad] = True

        x = self.attn(
            x,
            memory,
            tgt_key_padding_mask=attn_mask_q,
            memory_key_padding_mask=attn_mask_mem,
        )

        x = x[~attn_mask_q]

        totl = 0
        xnew = []
        for b, len in zip(range(B), lens_last_frame):
            current_batch = rearrange(
                x[totl : totl + len],
                "(n h w) f -> n f h w",
                h=self.pool_size[0],
                w=self.pool_size[1],
            )
            xnew.append(current_batch)
            totl += len

        x = torch.cat(xnew, dim=0)
        x = self.projector(x)

        if torch.isnan(x).any():
            raise ValueError("NaN detected")

        return x
