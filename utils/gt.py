from typing import Union


import cv2
import numpy as np
import torch

from utils.utils_scripts import int_scale, normalize
from utils.hmap_utils import get_hmap_patch, place
from utils.bboxes import draw_bbox


def get_gt(img, bboxes, dists, sigma, ds_stats, scale_factor=0.25, radius=4):
    """
    :param img: input image; numpy array with shape (H, W, 3)
    :param bboxes: list of bboxes, where each bbox is in the form `[x_min, y_min, x_max, y_max]`
    :param sigma: sigma of the 2D Gaussian centered in 'center'
    :param ds_stats: dictionary containing statistics of the training set, in particular:
        >> 'h_mean' and 'h_std' -> mean and standard deviation of the height of the bboxes
        >> 'w_mean' and 'w_std' -> mean and standard deviation of the width of the bboxes
    :param scale_factor: the output maps are smaller than the input image; this value is in range ]0, 1[
        >> scale_factor = output_map.H / input_img.H
    :return: output maps: center heatmap and radius map
        >> c_map:  center heatmap -> shape: (H*scale_factor, W*scale_factor)
        >> h_map:  bbox-height map -> shape: (H*scale_factor, W*scale_factor)
        >> w_map:  bbox-width map -> shape: (H*scale_factor, W*scale_factor)
    """
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()
    if isinstance(dists, torch.Tensor):
        dists = dists.tolist()
    # create heatmap patch
    side = int(round((sigma * 6.25)))
    side = side if side % 2 == 1 else side + 1  # patch side must be odd
    center = [1 + side // 2, 1 + side // 2]
    hmap_patch = get_hmap_patch(hmap_h=side, hmap_w=side, center_xy=center, sigma=sigma)

    # init center heatmap and H/W maps
    scaled_h = int_scale(img.shape[0], scale_factor=scale_factor)
    scaled_w = int_scale(img.shape[1], scale_factor=scale_factor)
    c_map = np.zeros((img.shape[0], img.shape[1]))
    h_map = np.zeros_like(c_map)
    w_map = np.zeros_like(c_map)
    d_map = np.zeros((scaled_h, scaled_w))
    m_map = np.zeros_like(d_map)

    for i in range(len(bboxes)):
        if dists:
            dist = dists[i]
        else:
            dist = 0

        if isinstance(bboxes[i], torch.Tensor):
            x_min, y_min, x_max, y_max = bboxes[i].tolist()
        else:
            x_min, y_min, x_max, y_max = bboxes[i]
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

        # skip bboxes whose center is outside the image
        if not (0 <= cx < img.shape[1]) or not (0 <= cy < img.shape[0]):
            continue

        # update center heatmap
        cx, cy = int_scale(cx, scale_factor=1), int_scale(cy, scale_factor=1)
        c_map = place(foreground=hmap_patch, background=c_map, x=cx, y=cy, mode="max")

        # update H map
        color = normalize(bbox_h, mean=ds_stats["h_mean"], std=ds_stats["h_std"])
        cv2.circle(
            h_map,
            center=(cx, cy),
            radius=radius,
            color=(color, color, color),
            thickness=-1,
        )  # *(1)
        # *(1) TODO: `h_map` circles have `radius=radius` (hardcoded)... maybe we want tho change this?

        # update W map
        color = normalize(bbox_w, mean=ds_stats["w_mean"], std=ds_stats["w_std"])
        cv2.circle(
            w_map,
            center=(cx, cy),
            radius=radius,
            color=(color, color, color),
            thickness=-1,
        )  # *(2)
        # *(2) TODO: `w_map` circles have `radius=radius` (hardcoded)... maybe we want tho change this?

        # update D map
        color = normalize(dist, mean=ds_stats["d_mean"], std=ds_stats["d_std"])
        cx, cy = (
            int_scale(cx, scale_factor=scale_factor),
            int_scale(cy, scale_factor=scale_factor),
        )
        cv2.circle(
            d_map,
            center=(cx, cy),
            radius=radius,
            color=(color, color, color),
            thickness=-1,
        )  # *(2)
        cv2.circle(
            m_map, center=(cx, cy), radius=radius, color=(1, 1, 1), thickness=-1
        )  # *(2)
        # *(2) TODO: `w_map` circles have `radius=radius` (hardcoded)... maybe we want tho change this?

    return c_map, h_map, w_map, d_map, m_map


def get_gt_cmap_only(
    img: Union[np.ndarray, torch.Tensor],
    bboxes: Union[torch.Tensor, list],
    sigma: float,
) -> np.ndarray:
    """Create a center heatmap from the given image and bboxes.

    Args:
        img (Union[np.ndarray, torch.Tensor]): input image
        bboxes (Union[torch.Tensor, list]): bbox coordinates (x_min, y_min, x_max, y_max) already scaled to the input image
        sigma (float): sigma value for the heatmap

    Returns:
        np.ndarray: center heatmap
    """
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()

    # create heatmap patch
    side = int(round((sigma * 6.25)))
    side = side + (side % 2 == 0)  # patch side must be odd
    center = [1 + side // 2] * 2
    hmap_patch = get_hmap_patch(hmap_h=side, hmap_w=side, center_xy=center, sigma=sigma)

    c_map = np.zeros((img.shape[0], img.shape[1]))

    for i in range(len(bboxes)):
        if isinstance(bboxes[i], torch.Tensor):
            x_min, y_min, x_max, y_max = bboxes[i].tolist()
        else:
            x_min, y_min, x_max, y_max = bboxes[i]

        cx, cy = int(x_min + x_max) // 2, int(y_min + y_max) // 2

        # skip bboxes whose center is outside the image
        if not (0 <= cx < img.shape[1]) or not (0 <= cy < img.shape[0]):
            continue

        # update center heatmap
        c_map = place(foreground=hmap_patch, background=c_map, x=cx, y=cy, mode="max")

    return c_map


def get_mask(img, mask_bboxes, scale_factor=0.25):
    """
    Get binary mask given a list of masking bounding boxes.
    :param img: image to which the mask refers
    :param mask_bboxes: list of bboxes representing the areas to be masked on the image
    :param scale_factor: the output binary mask is smaller than the input image; this value is in range ]0, 1[
        >> scale_factor = output_mask.H / input_img.H
    :return: binary mask with value 0 for the parts of the image that must be ignored and value 1 otherwise
    """

    scaled_h = int_scale(img.shape[0], scale_factor=scale_factor)
    scaled_w = int_scale(img.shape[1], scale_factor=scale_factor)

    mask = np.ones((scaled_h, scaled_w))

    for bbox in mask_bboxes:
        bbox = [int_scale(coord, scale_factor=scale_factor) for coord in bbox]
        mask = draw_bbox(mask, bbox, color=(0, 0, 0), thickness=-1)

    return mask
