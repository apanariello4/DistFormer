import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from scripts.noisy_bb import iou_pairs

from utils.plots import draw_bb


class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)"""

    def forward(self, vid: torch.Tensor, *args) -> torch.Tensor:
        if not args:
            return vid.permute(1, 0, 2, 3)
        return vid.permute(0, 3, 1, 2), *args


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor, *args) -> torch.Tensor:
        if not args:
            return vid.permute(1, 0, 2, 3)
        return vid.permute(1, 0, 2, 3), *args


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(
        self,
        image: torch.Tensor,
        targets: list[torch.Tensor],
        dists: list[torch.Tensor],
        good_idx: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.p:
            _, _, w = F.get_dimensions(image)
            image = F.hflip(image)

            for t in targets:
                if len(t) == 0:
                    continue
                t[:, [2, 0]] = t[:, [0, 2]].sub_(w).neg_()

        return image, targets, dists, good_idx


class Resize(transforms.Resize):
    def __init__(
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
    ):
        super().__init__(size, interpolation, max_size, antialias)

    def forward(self, img, targets, dists, good_idx):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        _, h, w = F.get_dimensions(img)
        h_scale = self.size[0] / h
        w_scale = self.size[1] / w
        img = F.resize(
            img, self.size, self.interpolation, self.max_size, self.antialias
        )
        scaling = torch.tensor([w_scale, h_scale, w_scale, h_scale])

        targets = [t.mul(scaling).to(torch.int32) if len(t) > 0 else t for t in targets]

        return img, targets, dists, good_idx


class RandomCrop(transforms.RandomCrop):
    PAD_MODES = ["constant", "edge", "reflect", "symmetric"]
    """Crop the given image at a random location, with a random size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        min_crop (float): Minimum crop size as a fraction of the original image size. The crop value
            is chosen randomly between min_crop and max_crop.
        max_crop (float): Maximum crop size as a fraction of the original image size. The crop value
            is chosen randomly between min_crop and max_crop.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is randomly chosen from constant, edge, reflect, symmetric.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(
        self,
        size: Tuple[int, int],
        crop_mode: str = "random",
        min_crop: float = 0.0,
        max_crop: float = 0.25,
        padding: Optional[List[int]] = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_modes: List[str] = PAD_MODES,
    ):
        super().__init__(
            size,
            padding,
            pad_if_needed,
            fill,
        )
        self.padding_modes = padding_modes
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.crop_mode = crop_mode

    def get_params(self, img: torch.Tensor) -> Tuple[int, int, int, int]:
        _, h, w = F.get_dimensions(img)

        tw = int(w * (1 - random.uniform(self.min_crop, self.max_crop)))
        th = int(h * (1 - random.uniform(self.min_crop, self.max_crop)))
        if w == tw and h == th:
            return 0, 0, h, w

        if self.crop_mode == "random":
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        elif self.crop_mode == "center":
            i = (h - th) // 2
            j = (w - tw) // 2

        return i, j, th, tw

    def forward(
        self,
        img: torch.Tensor,
        targets: List[torch.Tensor],
        dists: List[torch.Tensor],
        good_idx: List[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        # choose a random pad from self.padding_modes
        random_pad = random.choice(self.padding_modes)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, random_pad)

        # pad the width if needed
        if self.pad_if_needed and img.shape[-1] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.shape[-1], 0), self.fill, random_pad)
        # pad the height if needed
        if self.pad_if_needed and img.shape[-2] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.shape[-2]), self.fill, random_pad)

        i, j, h, w = self.get_params(img)

        img = F.crop(img, top=i, left=j, height=h, width=w)
        new_targets = []
        new_dists = []
        for n, (t, d) in enumerate(zip(targets, dists)):
            if len(t) == 0:
                new_targets.append(t)
                new_dists.append(d)
                continue

            t[:, [0, 2]] -= j
            t[:, [1, 3]] -= i

            # keep only boxes whose centers are in the crop
            mask = (
                (0 < t[:, 0] + t[:, 2])
                & (t[:, 0] + t[:, 2] <= w * 2)
                & (0 < t[:, 1] + t[:, 3])
                & (t[:, 1] + t[:, 3] <= h * 2)
            )

            t = t[mask]
            d = d[mask]

            good_idx[n] = good_idx[n] * mask

            t = torchvision.ops.clip_boxes_to_image(t, (h, w)).int()

            # t[:, [0, 2]] = t[:, [0, 2]].clamp(min=0, max=w)
            # t[:, [1, 3]] = t[:, [1, 3]].clamp(min=0, max=h)
            # t = t.int()

            new_targets.append(t)
            new_dists.append(d)

        return img, new_targets, new_dists, good_idx


class ConvertImageDtypeBB(transforms.ConvertImageDtype):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)

    def forward(self, img, targets, dists, good_idx):
        return F.convert_image_dtype(img, self.dtype), targets, dists, good_idx


class ComposeBB(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, targets, dists, good_idx):
        for t in self.transforms:
            img, targets, dists, good_idx = t(img, targets, dists, good_idx)
        return img, targets, dists, good_idx


class ToTensorBB(transforms.ToTensor):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(
        self, image: torch.Tensor, target, dists, good_idx
    ) -> Tuple[torch.Tensor, list, list, list]:
        image = torch.tensor(image) if isinstance(image, np.ndarray) else image

        dists = [torch.tensor(d, dtype=torch.float32) for d in dists]

        return image, target, dists, good_idx


class NoisyBB(nn.Module):
    def __init__(self, min_offset=0.0, max_offset=0.01, min_scale=0.9, max_scale=1.1):
        """
        Args:
            min_offset (float): minimum offset percentage of image size (0.0 - 1.0)
            max_offset (float): maximum offset percentage of image size (0.0 - 1.0)
            min_scale (float): minimum scale
            max_scale (float): maximum scale
        """
        super().__init__()
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, img, targets, dists, good_idx):
        _, h, w = F.get_dimensions(img)
        new_targets = []
        new_dists = []

        for n, (t, d) in enumerate(zip(targets, dists)):
            if len(t) == 0:
                new_targets.append(t)
                new_dists.append(d)
                continue

            offset = (
                torch.rand(len(t), 2) * (self.max_offset - self.min_offset)
                + self.min_offset
            ) * torch.tensor([w, h])
            scale = (
                torch.rand(len(t), 2) * (self.max_scale - self.min_scale)
                + self.min_scale
            )

            # apply offset to x coords
            t[:, [0, 2]] = t[:, [0, 2]] + offset[:, 0:1]
            # apply offset to y coords
            t[:, [1, 3]] = t[:, [1, 3]] + offset[:, 1:2]

            x_diff = t[:, 2] - t[:, 0]  # get width of box
            y_diff = t[:, 3] - t[:, 1]  # get height of box
            scale = (1 - scale) / 2
            # scale zooms in/out from the center of the box
            t[:, 0] = torch.where(
                scale[:, 0] > 1,
                t[:, 0] + (x_diff * scale[:, 0]),
                t[:, 0] - (x_diff * scale[:, 0]),
            )
            t[:, 1] = torch.where(
                scale[:, 1] > 1,
                t[:, 1] + (y_diff * scale[:, 1]),
                t[:, 1] - (y_diff * scale[:, 1]),
            )
            t[:, 2] = torch.where(
                scale[:, 0] > 1,
                t[:, 2] - (x_diff * scale[:, 0]),
                t[:, 2] + (x_diff * scale[:, 0]),
            )
            t[:, 3] = torch.where(
                scale[:, 1] > 1,
                t[:, 3] - (y_diff * scale[:, 1]),
                t[:, 3] + (y_diff * scale[:, 1]),
            )

            # keep only boxes whose centers are in the crop
            mask = (
                (0 < t[:, 0] + t[:, 2])
                & (t[:, 0] + t[:, 2] <= w * 2)
                & (0 < t[:, 1] + t[:, 3])
                & (t[:, 1] + t[:, 3] <= h * 2)
            )

            t = t[mask]
            d = d[mask]

            good_idx[n] = good_idx[n] * mask

            t = torchvision.ops.clip_boxes_to_image(t, (h, w)).int()

            new_targets.append(t)
            new_dists.append(d)

        return img, new_targets, new_dists, good_idx


class NoisyBBIouPreserving(nn.Module):
    def __init__(self, min_iou) -> None:
        super().__init__()
        self.min_iou = min_iou

    def forward(self, img, targets, dists, good_idx):
        original_bboxes = targets.copy()
        new_targets = []

        for n, bbox in enumerate(targets):
            if len(bbox) == 0:
                new_targets.append(bbox)
                continue

            bbox = bbox.cpu().detach().numpy()

            upper_bound_inside_box = (1 - np.sqrt(self.min_iou)) / 2
            upper_bound_outside_box = 0.5 * (1 / np.sqrt(self.min_iou) - 1)
            max_eps = min(upper_bound_inside_box, upper_bound_outside_box)

            epsilons = np.random.uniform(-max_eps, max_eps, 4 * bbox.shape[0]).reshape(
                -1, 4
            )
            w = bbox[:, 2] - bbox[:, 0]
            h = bbox[:, 3] - bbox[:, 1]
            coord_shift_vals = np.array([h, h, w, w], dtype=np.float32).T * epsilons
            bb_top_shift, bb_bot_shift, bb_left_shift, bb_right_shift = (
                coord_shift_vals.T
            )

            bbox[:, 0] += bb_left_shift
            bbox[:, 1] += bb_top_shift
            bbox[:, 2] += bb_right_shift
            bbox[:, 3] += bb_bot_shift

            # Just make sure that we did not surpass the IoU threshold (dataset_params['min_iou_bb_wiggling'])
            iou_val = iou_pairs(bbox, original_bboxes[n].cpu().detach().numpy())
            assert (
                iou_val.min() > self.min_iou
            ), f"Found iou_val.min() = {iou_val.min()} < {self.min_iou} = min_iou"

            new_targets.append(torch.tensor(bbox, dtype=torch.float32))

        return img, new_targets, dists, good_idx


if __name__ == "__main__":
    batch = torch.load("img.pth")

    img, targets, dists = batch
    to_tensor = ComposeBB(
        [
            ToTensorBB(),
            ConvertBHWCtoBCHW(),
            ConvertImageDtypeBB(torch.float32),
        ]
    )

    img, targets, dists = to_tensor(img, targets, dists)

    img, targets_new, dists = RandomCrop((360, 640), min_crop=0.4, max_crop=0.7)(
        img, targets, dists
    )
    img, targets_new, dists = Resize((360, 640))(img, targets_new, dists)
    img = draw_bb(img[0], targets_new[0], dists[0], "debug")
    print("saved debug img")
