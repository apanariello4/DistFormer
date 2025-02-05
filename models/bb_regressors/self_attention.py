from typing import Optional

import torch
import torch.nn as nn

from models.bb_regressors.pos_emedding import PosEmbeddingPatches


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        norm: bool = False,
        embed: Optional[str] = None,
        init: bool = False,
        num_patches: int = 4,
    ) -> None:
        super().__init__()
        self.embed = embed
        if embed == "positional":
            self.embedding = PosEmbeddingPatches(embed_dim, num_patches)

        self.sa = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.ffa = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        if init:
            self.init_transformer()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask=None,
        attn_mask=None,
        bboxes=None,
        H=None,
        W=None,
    ) -> torch.Tensor:
        res = x
        if self.embed == "positional":
            x = self.embedding(x)

        x = (
            res
            + self.sa(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[
                0
            ]
        )
        x = self.dropout(self.norm1(x))
        x = res + self.ffa(x)
        x = self.dropout(self.norm2(x))
        return x

    def init_transformer(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_()
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
