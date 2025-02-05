import argparse
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torchvision
from einops import rearrange, reduce

from models.bb_regressors.self_attention import SelfAttention
from models.gcn import GCN, StackedGATv2
from models.mae import MAE

from .regressor_utils import flatten_bboxes, get_last_frame, split_by_frame
from .vit_encoder import get_enc_dec

warnings.filterwarnings("ignore", category=UserWarning)


class MAERegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        args: argparse.Namespace,
        output_dim: int = 512,
        pool_size: int = 4,
        dropout: float = 0.0,
        roi_op: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.roi_op = roi_op if roi_op is not None else torchvision.ops.roi_pool
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_heads = 4
        self.encoder, decoder = get_enc_dec(args)
        self.hidden_dim = self.encoder.encoder_dim
        self.global_encoder = args.mae_global_encoder
        self.classify_anchor = args.classify_anchor
        self.avg_pool_post_roi = args.avg_pool_post_roi

        if self.global_encoder == "self_attention":
            self.self_attention = SelfAttention(
                num_heads=self.num_heads,
                norm=True,
                embed_dim=self.hidden_dim,
                embed=None,
                init=False,
                dropout=dropout,
            )
        elif self.global_encoder == "gcn":
            self.self_attention = GCN(self.encoder.encoder_dim, 512)
        elif self.global_encoder == "gat":
            self.self_attention = StackedGATv2(
                self.encoder.encoder_dim, 2, 4, self.encoder.encoder_dim, 0.2
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

        patch_size = args.mae_patch_size
        pool_window = (self.pool_size**2 if isinstance(self.pool_size, int) else self.pool_size[0] * self.pool_size[1])
        pool_window = (pool_window // (self.avg_pool_post_roi**2) if self.avg_pool_post_roi else pool_window)
        num_patches = pool_window // (patch_size**2)
        if args.mae_target == "input":
            p0 = (self.pool_size[0] // self.avg_pool_post_roi if self.avg_pool_post_roi else self.pool_size[0])
            p1 = (self.pool_size[1] // self.avg_pool_post_roi if self.avg_pool_post_roi else self.pool_size[1])
            token_to_crop_size = (args.mae_crop_size[0] // p0, args.mae_crop_size[1] // p1)

        self.mae = MAE(
            args=args,
            input_dim=self.hidden_dim,
            num_patches=num_patches,
            patch_size=args.mae_patch_size,
            detach=args.mae_detach,
            masking_ratio=args.mae_masking,
            mae_eval_masking=args.mae_eval_masking,
            mae_eval_masking_ratio=args.mae_eval_masking_ratio,
            encoder=self.encoder,
            decoder=decoder,
            token_to_crop_size=token_to_crop_size
            if args.mae_target == "input"
            else None,
            activation=args.mae_activation,
            min_brightness=args.mae_min_brightness,
        )

        self.output = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim),)

    def forward(
        self,
        x,
        bboxes: list,
        scale: Optional[float] = None,
        crops: Optional[torch.Tensor] = None,
        save_crops: bool = False,
        small_bboxes: Optional[torch.Tensor] = None,
        gt_anchors: Optional[torch.Tensor] = None,
        encode_only: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """input shape: (B, C, T, H, W),
        bboxes shape: (B, N, 4)"""
        B, C, T, H, W = x.shape

        x = rearrange(x, "b c t h w -> (b t) c h w")
        flattened_bboxes = flatten_bboxes(bboxes)
        n_objects_per_frame = [len(b) for b in flattened_bboxes]

        roi_feat = self.get_roi_features(x, flattened_bboxes, B * T, scale)

        objects = split_by_frame(roi_feat, n_objects_per_frame)

        objects_batches, lens = get_last_frame(objects, B, T)

        x = torch.cat(objects_batches, dim=0)

        x = self.standard_projector(x)

        if self.training and not encode_only:
            losses, encoded_feat = self.mae(
                x,
                return_encoder=True,
                crops=crops,
                save_crops=save_crops,
                bboxes=bboxes,
                small_bboxes=small_bboxes,
                gt_anchors=gt_anchors,
            )
        else:
            if self.training:
                # this means that we disabled mae
                self.mae.masking_ratio = 0.0
            encoded_feat = self.mae(x, encode_only=True, crops=crops)
            losses = {}

        x = reduce(encoded_feat, "(b t) tokens c -> (b t) c", "mean", t=T)

        if self.global_encoder == "self_attention":
            x = list(torch.split(x, lens, dim=0))

            x = [x_ for x_ in x if len(x_) > 0]

            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)

            padding_mask, _ = self.get_masks(lens=lens, device=x.device)

            last_frame_bbox = [b[-1] for b in bboxes if len(b) > 0]

            x = self.self_attention(x, key_padding_mask=padding_mask, bboxes=last_frame_bbox, H=H, W=W)
            x = x[padding_mask == False]

        elif self.global_encoder in ("gcn", "gat"):
            adj = torch.block_diag(*[torch.ones((len, len)) for len in lens])
            adj = torch_geometric.utils.dense_to_sparse(adj)[0].to(x.device)
            x = self.self_attention(x, adj)

        elif self.global_encoder == "none":
            pass

        out = self.output(x)

        return (out, losses) if self.training else out

    def get_roi_features(self, x, bboxes: list, n_frames: int, scale: float) -> torch.Tensor:
        roi_indices = torch.repeat_interleave(torch.arange(0, n_frames), torch.tensor([len(b) for b in bboxes], requires_grad=False)).to(x.device)
        roi_bboxes = torch.cat(bboxes).to(x.device)
        rois = torch.cat([roi_indices[:, None].float(), roi_bboxes], dim=1)

        roi_feat = self.roi_op(input=x, boxes=rois.type_as(x), output_size=self.pool_size, spatial_scale=scale)
        return (F.avg_pool2d(roi_feat, kernel_size=self.avg_pool_post_roi) if self.avg_pool_post_roi else roi_feat)

    def get_masks(self, lens: list, device: torch.device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generates padding and attention masks for self-attention.

        Args:
            lens (list): list of lengths of each sequence
            device (torch.device): device to put the masks on
        """
        lens = torch.tensor(lens, device=device)
        padding_mask = (torch.arange(max(lens), device=device)[None, :] >= lens[lens.nonzero(as_tuple=True)[0]][:, None])

        return padding_mask, None
