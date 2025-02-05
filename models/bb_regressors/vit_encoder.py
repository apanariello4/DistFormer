import argparse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block

from .vit_fb import vit_base_patch16


class CustomVit(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        layers = args.mae_encoder_layers
        encoder = torchvision.models.vit_b_16(
            weights="DEFAULT" if args.mae_vit_pretrained else None
        ).encoder

        self.layers = encoder.layers[len(encoder.layers) - layers :]
        self.ln = encoder.ln
        self.encoder_dim = args.mae_encoder_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.layers(x))


class CustomVitReverse(nn.Module):
    """
    First mae_vit_layers excluding first layer used for encoder,
    last mae_vit_layers used for decoder but reversed.
    """

    def __init__(self, args: argparse.Namespace, mode: str) -> None:
        super().__init__()
        layers = args.mae_encoder_layers
        encoder = torchvision.models.vit_b_16(
            weights="DEFAULT" if args.mae_vit_pretrained else None
        ).encoder
        if mode == "encoder":
            self.layers = encoder.layers[:layers]
        elif mode == "decoder":
            self.layers = encoder.layers[layers:][::-1]
        self.ln = encoder.ln
        self.encoder_dim = args.mae_encoder_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.layers(x))


class CustomAngelVit(nn.Module):
    """
    ViT with mae_encoder_layers used for encoder, mae_decoder_layers used for decoder.
    """

    def __init__(self, args: argparse.Namespace, mode: str) -> None:
        super().__init__()
        self.mode = mode
        vit = torchvision.models.vit_b_16(
            weights="DEFAULT" if args.mae_vit_pretrained else None
        )
        encoder = vit.encoder
        self.use_cls_token = args.classify_anchor
        if mode == "decoder":
            self.layers = encoder.layers[-args.mae_decoder_layers :]
        elif mode == "encoder":
            self.layers = encoder.layers[
                -(
                    args.mae_decoder_layers + args.mae_encoder_layers
                ) : -args.mae_decoder_layers
            ]
            if self.use_cls_token:
                self.class_token = vit.class_token

        self.ln = encoder.ln
        self.encoder_dim = args.mae_encoder_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "class_token"):
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
        return self.ln(self.layers(x))


class CustomFBVit(nn.Module):
    def __init__(self, args: argparse.Namespace, mode: str) -> None:
        super().__init__()
        self.use_cls_token = args.classify_anchor
        if mode == "encoder":
            encoder = vit_base_patch16()
            del encoder.head

            if args.mae_vit_pretrained:
                path = Path("data/mae_pretrain_vit_base.pth")
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
                    print("downloading pretrained weights for mae")
                    urllib.request.urlretrieve(url, path)

                encoder.load_state_dict(
                    torch.load("data/mae_pretrain_vit_base.pth")["model"]
                )
                print("Loaded pretrained ViT weights from fb")

            for module in ("patch_embed", "pos_drop", "pre_logits"):
                if module in encoder._modules:
                    del encoder._modules[module]

                self.layers = encoder.blocks[-args.mae_encoder_layers :]
            self.ln = encoder.norm
            if self.use_cls_token:
                self.class_token = encoder.cls_token
        elif mode == "decoder":
            encoder = nn.ModuleList(
                [
                    Block(
                        args.mae_encoder_dim,
                        num_heads=8,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=nn.LayerNorm,
                    )
                    for i in range(args.mae_decoder_layers)
                ]
            )
            self.layers = encoder
            self.ln = nn.LayerNorm(args.mae_encoder_dim)
        self.encoder_dim = args.mae_encoder_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "class_token"):
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

        for blk in self.layers:
            x = blk(x)
        return self.ln(x)


def get_enc_dec(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.nn.Module]:
    if args.mae_version == 1:
        encoder = CustomVit(args)
        decoder_dim = encoder.encoder_dim
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=8,
            batch_first=True,
            dim_feedforward=decoder_dim * 4,
            activation=args.mae_activation,
        )
        decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=args.mae_decoder_layers
        )
    elif args.mae_version == 2:
        encoder = CustomAngelVit(args, mode="encoder")
        decoder = CustomAngelVit(args, mode="decoder")
    elif args.mae_version == 3:
        encoder = CustomVitReverse(args, mode="encoder")
        decoder = CustomVitReverse(args, mode="decoder")
    elif args.mae_version == 4:
        encoder = CustomFBVit(args, mode="encoder")
        decoder = CustomFBVit(args, mode="decoder")

    return encoder, decoder
