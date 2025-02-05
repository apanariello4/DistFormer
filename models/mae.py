from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
from einops import rearrange, repeat
from PIL import Image

import wandb
from models.losses.focal_frequency_loss import FocalFrequencyLoss


def pair(t: Union[int, tuple]) -> tuple:
    return tuple(t) if isinstance(t, (tuple, list)) else (t, t)


class MAE(nn.Module):
    def __init__(
        self,
        args,
        input_dim: int,
        num_patches: int,
        encoder_dim: int = None,
        decoder_dim: int = None,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        patch_size: Union[int, tuple] = 1,
        masking_ratio: float = 0.75,
        mae_eval_masking: bool = False,
        mae_eval_masking_ratio: float = 0.75,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_heads: int = 8,
        token_to_crop_size: Optional[Tuple[int, int]] = None,
        norm_pix_loss: bool = False,
        activation: str = "relu",
        min_brightness: float = 0,
        **kwargs,
    ):
        super().__init__()

        assert not (
            encoder is encoder_dim is None
        ), "encoder or encoder_dim must be specified"

        patch_height, patch_width = pair(patch_size)
        self.p1 = patch_height
        self.p2 = patch_width
        self.masking_ratio = masking_ratio
        self.mae_eval_masking = mae_eval_masking
        self.mae_eval_masking_ratio = mae_eval_masking_ratio
        self.patch_dim = patch_height * patch_width * input_dim
        self.token_to_crop_size = token_to_crop_size
        self.norm_pix_loss = norm_pix_loss
        self.roi_window = args.pool_size

        self.weight_night = args.mae_weight_night
        self.weight_small_bboxes = args.mae_weight_small_bboxes
        self.min_brightness = min_brightness

        self.loss = {
            "mse": nn.MSELoss(reduction="none"),
            "focal": FocalFrequencyLoss(),
        }[args.mae_loss]
        self.loss_no_masked = args.mae_loss_no_masked
        self.alpha_no_masked = args.mae_alpha_loss_no_masked
        self.classify_anchor = args.classify_anchor

        if encoder:
            if isinstance(encoder, nn.TransformerEncoder):
                encoder_dim = encoder.layers[0].self_attn.embed_dim
            else:
                encoder_dim = encoder.encoder_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, encoder_dim))

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        if encoder:
            self.encoder = encoder
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=encoder_heads,
                batch_first=True,
                activation=activation,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=encoder_layers
            )

        if decoder:
            self.decoder = decoder
            if isinstance(decoder, nn.TransformerEncoder):
                decoder_dim = decoder.layers[0].self_attn.embed_dim
            elif isinstance(decoder, nn.ModuleList):
                decoder_dim = decoder[0].attn.qkv.in_features

            else:
                decoder_dim = decoder.encoder_dim
        else:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                batch_first=True,
                dim_feedforward=decoder_dim * 4,
                activation=activation,
            )
            self.decoder = nn.TransformerEncoder(
                decoder_layer, num_layers=decoder_layers
            )

        self.decoder_dim = decoder_dim

        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )

        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)

        if self.classify_anchor:
            self.anchor_head = nn.Linear(decoder_dim, 9)
            self.anchor_head.weight.data.zero_()
            self.anchor_head.bias.data.zero_()
            self.classification_loss = nn.CrossEntropyLoss(reduction="none")

        if self.token_to_crop_size:
            self.to_pixels = nn.Linear(
                decoder_dim, self.token_to_crop_size[0] * self.token_to_crop_size[1] * 3
            )

        else:
            self.to_pixels = nn.Linear(decoder_dim, self.patch_dim)

    def adaptive(
        self,
        mask_tokens: torch.Tensor,
        gt_anchors: list[torch.Tensor],
        crops: list[torch.Tensor],
        small_bboxes,
        batch_range,
        masked_indices,
        unmasked_indices,
        cls,
        decoded_tokens,
    ):
        all_pred_pixels, all_targets, all_weights = [], [], []
        losses = defaultdict(list)
        for i, anchor in enumerate(self.rounded_anchors):
            # current_bboxes = torch.cat(sum(gt_anchors, [])) == i
            current_bboxes = torch.cat(tuple(chain.from_iterable(gt_anchors))) == i

            if current_bboxes.sum() == 0:
                continue

            pred_pixels = self.to_pixels[str(i)](mask_tokens[current_bboxes])
            all_pred_pixels.append(pred_pixels)
            weight = torch.ones(pred_pixels.shape[0], device=mask_tokens.device)
            # pred_pixels is [b, num_masked, patch_dim] or [b, num_masked, pixels_one_patch]

            # current_crops = sum(crops, [])
            current_crops = list(chain.from_iterable(crops))
            current_crops = [
                current_crops[i].unsqueeze(0)
                for i in range(len(current_crops))
                if current_bboxes[i]
            ]
            current_crops = torch.cat(current_crops, dim=0)

            crops_brightness = current_crops.max(dim=1, keepdim=True)[0].mean(
                dim=(1, 2, 3)
            )
            small_bboxes_mask = ~torch.cat(small_bboxes)[current_bboxes]

            night_mask = crops_brightness < self.min_brightness
            weight[night_mask] = self.weight_night
            weight[small_bboxes_mask] = self.weight_small_bboxes
            all_weights.append(weight)
            # original_crop = current_crops
            H = anchor[0].item() // self.roi_window[0]
            W = anchor[1].item() // self.roi_window[1]
            current_crops = rearrange(
                current_crops, "b c (h H) (w W) -> b (h w) (c H W) ", H=H, W=W
            )
            # target = current_crops[masked_indices[current_bboxes]]
            target = torch.gather(
                current_crops,
                1,
                masked_indices[current_bboxes][:, :, None].expand(
                    -1, -1, current_crops.shape[-1]
                ),
            )
            all_targets.append(target)
            # if save_crops:
            #     self.save_crops(x, decoded_tokens, original_crop, night_mask=night_mask,
            #                     small_bboxes_mask=small_bboxes_mask)

            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            # for i, (pred, gt, weight) in enumerate(zip(all_pred_pixels, all_targets, all_weights)):
            recon_loss = self.loss(input=pred_pixels, target=target)
            if recon_loss.ndim > 1:
                weight_masked = weight[:, None, None].expand_as(recon_loss)
                recon_loss_num, recon_loss_den = (
                    (recon_loss * weight_masked).sum((1, 2)),
                    weight_masked.sum((1, 2)),
                )
                losses["mae_loss_num"].append(recon_loss_num)
                losses["mae_loss_den"].append(recon_loss_den)

            if self.loss_no_masked:
                # target_clean = current_crops[batch_range[current_bboxes], masked_indices[current_bboxes]]
                target_clean = torch.gather(
                    current_crops,
                    1,
                    unmasked_indices[current_bboxes][:, :, None].expand(
                        -1, -1, current_crops.shape[-1]
                    ),
                )
                pred_pixels_clean = self.to_pixels[str(i)](
                    decoded_tokens[
                        batch_range[current_bboxes], unmasked_indices[current_bboxes]
                    ]
                )
                recon_loss_clean = self.loss(
                    input=pred_pixels_clean, target=target_clean
                )
                weight_unmasked = weight[:, None, None].expand_as(recon_loss_clean)
                recon_loss_clean = recon_loss_clean * weight_unmasked
                recon_loss_clean_num, recon_loss_clean_den = (
                    recon_loss_clean.sum((1, 2)),
                    weight_unmasked.sum((1, 2)),
                )
                losses["mae_loss_unmasked_num"].append(recon_loss_clean_num)
                losses["mae_loss_unmasked_den"].append(recon_loss_clean_den)

            if self.classify_anchor:
                pred_anchor = self.anchor_head(cls)
                # anchor_loss = self.classification_loss(pred_anchor, torch.cat(sum(gt_anchors, [])).to(pred_anchor.device))
                anchor_loss = self.classification_loss(
                    pred_anchor,
                    torch.cat(tuple(chain.from_iterable(gt_anchors))).to(
                        pred_anchor.device
                    ),
                )
                losses["anchor_loss"].append(anchor_loss)

        tot_losses = {}
        if len(losses["mae_loss_num"]) > 0:
            recon_loss_num = torch.cat(losses["mae_loss_num"]).sum(0)
            recon_loss_den = torch.cat(losses["mae_loss_den"]).sum(0)
            tot_losses["mae_loss"] = recon_loss_num / (recon_loss_den + 1e-9)
        if len(losses["mae_loss_unmasked_num"]) > 0:
            recon_loss_clean_num = torch.cat(losses["mae_loss_unmasked_num"]).sum(0)
            recon_loss_clean_den = torch.cat(losses["mae_loss_unmasked_den"]).sum(0)
            tot_losses["mae_loss_unmasked"] = (
                recon_loss_clean_num / (recon_loss_clean_den + 1e-9)
            ) * self.alpha_no_masked
        if len(losses["anchor_loss"]) > 0:
            tot_losses["anchor_loss"] = torch.stack(losses["anchor_loss"]).mean()

        return tot_losses

    def forward(
        self,
        x: torch.Tensor,
        return_encoder: bool = False,
        encode_only: bool = False,
        crops: Optional[torch.Tensor] = None,
        save_crops: bool = False,
        small_bboxes: Optional[torch.Tensor] = None,
        gt_anchors: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        x: [b, c, h, w]
        crops: [num_bboxes, 3, H_CROP, W_CROP]
        """
        device = x.device

        # get patches
        patches = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.p1, p2=self.p2
        )
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.to_patch_embedding(patches)
        tokens = tokens + self.pos_embedding[:, 1 : (num_patches + 1)]

        if encode_only and not self.training and not self.mae_eval_masking:
            return (
                self.encoder(tokens)[:, 1:]
                if self.classify_anchor
                else self.encoder(tokens)
            )

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        if not self.training and self.mae_eval_masking:
            num_masked = int(self.mae_eval_masking_ratio * num_patches)
        else:
            num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder(tokens)
        if self.classify_anchor:
            cls, encoded_tokens = encoded_tokens[:, 0], encoded_tokens[:, 1:]

        if encode_only:
            return encoded_tokens

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(
            unmasked_indices
        )

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device
        )
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]

        pred_pixels = self.to_pixels(mask_tokens)

        weight = torch.ones(pred_pixels.shape[0], device=device)
        # pred_pixels is [b, num_masked, patch_dim] or [b, num_masked, pixels_one_patch]
        if self.token_to_crop_size:
            # detect crops at night
            crops_brightness = crops.max(dim=1, keepdim=True)[0].mean(dim=(1, 2, 3))
            small_bboxes_mask = ~torch.concatenate(small_bboxes)

            night_mask = crops_brightness < self.min_brightness
            weight[night_mask] = self.weight_night
            weight[small_bboxes_mask] = self.weight_small_bboxes

            original_crop = crops
            crops = rearrange(
                crops,
                "b c (h H) (w W) -> b (h w) (c H W) ",
                H=self.token_to_crop_size[0],
                W=self.token_to_crop_size[1],
            )
            target = crops[batch_range, masked_indices]
            if save_crops:
                self.save_crops(
                    x,
                    decoded_tokens,
                    original_crop,
                    night_mask=night_mask,
                    small_bboxes_mask=small_bboxes_mask,
                )
        else:
            target = masked_patches

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        losses = {}
        recon_loss = self.loss(input=pred_pixels, target=target)
        if recon_loss.ndim > 1:
            weight_masked = weight[:, None, None].expand_as(recon_loss)
            recon_loss = (recon_loss * weight_masked).sum() / (
                weight_masked.sum() + 1.0e-6
            )
            losses["mae_loss"] = recon_loss

        if self.loss_no_masked:
            target_clean = crops[batch_range, unmasked_indices]
            pred_pixels_clean = self.to_pixels(
                decoded_tokens[batch_range, unmasked_indices]
            )
            recon_loss_clean = self.loss(input=pred_pixels_clean, target=target_clean)
            weight_unmasked = weight[:, None, None].expand_as(recon_loss_clean)
            recon_loss_clean = recon_loss_clean * weight_unmasked * self.alpha_no_masked
            recon_loss_clean = recon_loss_clean.sum() / (weight_unmasked.sum() + 1.0e-6)
            losses["mae_loss_unmasked"] = recon_loss_clean

        if self.classify_anchor:
            pred_anchor = self.anchor_head(cls)
            # anchor_loss = self.classification_loss(pred_anchor, torch.cat(sum(gt_anchors, [])).to(device)).mean()
            anchor_loss = self.classification_loss(
                pred_anchor,
                torch.cat(tuple(chain.from_iterable(gt_anchors))).to(device),
            ).mean()
            losses["anchor_loss"] = anchor_loss

        return (losses, encoded_tokens) if return_encoder else losses

    @torch.no_grad()
    def save_crops(
        self, x, decoded_tokens, original_crop, night_mask=None, small_bboxes_mask=None
    ):
        h, w = x.shape[-2:]
        all_pixels = self.to_pixels(decoded_tokens).detach().cpu()
        all_pixels = rearrange(
            all_pixels,
            "b (h w) (c H W) -> b c (h H) (w W)",
            H=self.token_to_crop_size[0],
            W=self.token_to_crop_size[1],
            h=h,
            w=w,
        )
        to_print = torch.cat(
            [original_crop, all_pixels.type_as(original_crop).to(original_crop.device)],
            dim=-1,
        )
        # draw red rectangle around night indices
        if night_mask is not None:
            for i in range(len(night_mask)):
                if night_mask[i]:
                    to_print[i][0, 0 : to_print[0].shape[1], 0] = 1
                    to_print[i][0, 0 : to_print[0].shape[1], -1] = 1
                    to_print[i][0, 0, 0 : to_print[0].shape[2]] = 1
                    to_print[i][0, -1, 0 : to_print[0].shape[2]] = 1

        if small_bboxes_mask is not None:
            for i in range(len(small_bboxes_mask)):
                if small_bboxes_mask[i]:
                    to_print[i][1, 0 : to_print[0].shape[1], 0] = 1
                    to_print[i][1, 0 : to_print[0].shape[1], -1] = 1
                    to_print[i][1, 0, 0 : to_print[0].shape[2]] = 1
                    to_print[i][1, -1, 0 : to_print[0].shape[2]] = 1

        path = Path("qualitatives", "predicted_crop.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(to_print, path)
        if wandb.run:
            wandb_img = (
                torchvision.utils.make_grid(to_print).permute(1, 2, 0).cpu().numpy()
            )
            wandb_img = Image.fromarray((wandb_img * 255).astype(np.uint8))
            wandb.log({"Predicted Crops": wandb.Image(wandb_img)})
