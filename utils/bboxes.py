from typing import Sequence, Union

import cv2
import numpy as np
import torch
import torchvision.ops as ops
import torchvision.transforms as transforms

from utils.utils_scripts import colors, to_color

Number = Union[int, float]
BBox = Sequence[Number]
CRBBox = Sequence[Number]


def resized_crops(batched_bboxes, original_image, crop_size, threshold_w, threshold_h):
    crops_list = []
    small_bboxes_list = []
    for i in range(original_image.shape[0]):
        crops = []
        bboxes = (
            batched_bboxes[i][-1]
            if len(batched_bboxes[i][-1].shape) != 3
            else batched_bboxes[i][-1][0]
        )
        small_bboxes = torch.ones(
            len(bboxes), dtype=torch.bool, device=original_image.device
        )
        for j, xyxy in enumerate(bboxes):
            x1, y1, _, _, h, w = get_bb_dimensions(original_image, xyxy)
            if w < threshold_w or h < threshold_h:
                small_bboxes[j] = False
            crop = transforms.functional.crop(original_image[i, :3, -1], y1, x1, h, w)
            crop = transforms.functional.resize(crop, crop_size, antialias=None)
            crops.append(crop)
        crops = (
            torch.stack(crops, dim=0)
            if crops
            else torch.empty((0, 3, *crop_size)).to(original_image.device)
        )
        crops_list.append(crops)
        small_bboxes_list.append(small_bboxes)
    return torch.concatenate(crops_list, dim=0), small_bboxes_list


def round_anchor(anchor, multiple):
    if isinstance(multiple, int):
        multiple = torch.tensor([multiple, multiple])
    elif isinstance(multiple, (tuple, list)):
        multiple = torch.tensor(multiple)
    if isinstance(anchor, np.ndarray):
        anchor = torch.tensor(anchor)
    return (torch.round(anchor / multiple) * multiple).int()


def adaptive_resized_crops(
    batched_bboxes,
    original_image,
    threshold_w,
    threshold_h,
    rounded_anchors,
    gt_anchors,
):
    crops_list = []
    small_bboxes_list = []
    B = original_image.shape[0]
    for i in range(B):
        crops = []
        bboxes = (
            batched_bboxes[i][-1]
            if len(batched_bboxes[i][-1].shape) != 3
            else batched_bboxes[i][-1][0]
        )
        small_bboxes = torch.ones(
            len(bboxes), dtype=torch.bool, device=original_image.device
        )
        for j, xyxy in enumerate(bboxes):
            x1, y1, x2, y2, h, w = get_bb_dimensions(original_image, xyxy)
            if w < threshold_w or h < threshold_h:
                small_bboxes[j] = False
            nn_anchor = rounded_anchors[gt_anchors[i][-1][j]]
            crop = transforms.functional.crop(original_image[i, :3, -1], y1, x1, h, w)
            crop = transforms.functional.resize(crop, nn_anchor, antialias=None)
            crops.append(crop)
        crops_list.append(crops)
        small_bboxes_list.append(small_bboxes)
    return crops_list, small_bboxes_list


def get_bb_dimensions(original_image, xyxy):
    x1, y1, x2, y2 = xyxy
    if x2 - x1 == 0:
        x1 = max(0, x1 - 1)
        x2 = min(original_image.shape[-1], x2 + 1)
    if y2 - y1 == 0:
        y1 = max(0, y1 - 1)
        y2 = min(original_image.shape[-2], y2 + 1)
    h, w = y2 - y1, x2 - x1
    return x1, y1, x2, y2, h, w


def bbox_nms(bboxes, confidences, iou_th):
    if len(bboxes) == 0:
        return bboxes, confidences

    idxs = ops.nms(
        boxes=torch.tensor(bboxes, dtype=torch.float),
        scores=torch.tensor(confidences, dtype=torch.float),
        iou_threshold=iou_th,
    )

    return bboxes[idxs], confidences[idxs]


def iou(bbox_a, bbox_b):
    """
    Intersection over Union (IoU) between 2 input bounding boxes: `bbox_a` and `bbox_b`
    :param bbox_a: first bbox in the form [x_min, y_min, x_max, y_max]
    :param bbox_b: second bbox in the form [x_min, y_min, x_max, y_max]
    :return: intersection over union
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    inter_x_min = max(bbox_a[0], bbox_b[0])
    inter_y_min = max(bbox_a[1], bbox_b[1])
    inter_x_max = min(bbox_a[2], bbox_b[2])
    inter_y_max = min(bbox_a[3], bbox_b[3])

    # compute the area of intersection rectangle
    inter_area = abs(
        max((inter_x_max - inter_x_min, 0)) * max((inter_y_max - inter_y_min), 0)
    )
    if inter_area == 0:
        return 0

    # compute the area of both input bboxes
    bbox_a_area = abs((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    bbox_b_area = abs((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

    # compute the intersection over union
    return inter_area / float(bbox_a_area + bbox_b_area - inter_area)


def draw_bbox(img, bbox, color=colors.FLAT_BLUE.value, thickness=2, alpha=0.0):
    """
    Draw bounding box `bbox` on a copy of the input image `img`.
    :param img: input image
    :param bbox: bounding box you want to draw; bbox is in the form [x_min, y_min, x_max, y_max]
    :param color: RGB color of the bbox rectangole
    :param thickness: thickness of the bbox rectangle
    :param alpha: alpha of the area of the bbox rectangle
    :return: copy of the input image with the bbox drawn on it
    """
    color = to_color(color, colorspace="RGB")
    p1 = (int(round(bbox[0])), int(round(bbox[1])))
    p2 = (int(round(bbox[2])), int(round(bbox[3])))
    if alpha > 0:
        x = cv2.rectangle(img.copy(), p1, p2, color=color, thickness=-1)
        x = ((1 - alpha) * img.astype(np.float) + alpha * x.astype(np.float)).astype(
            np.uint8
        )
    else:
        x = img.copy()
    return cv2.rectangle(x, p1, p2, color, thickness)


def get_h(bbox):
    return bbox[3] - bbox[1]


def get_w(bbox):
    return bbox[2] - bbox[0]


def post_process_half_bboxes(bboxes_half, confs_half, min_h):
    if len(bboxes_half) == 0:
        return bboxes_half, confs_half

    bboxes_half = 2 * bboxes_half
    filter_idx = bboxes_half[:, 3] - bboxes_half[:, 1] > min_h
    return np.array(bboxes_half)[filter_idx], np.array(confs_half)[filter_idx]


def bbox_noise(bbox, cnf, mode, tid):
    """
    Augment the input bbox based on the track_id (tid) (every tid has its own noise parameters)
    :param bbox: input bbox (ltbr)
    :param cnf: configuration
    :param mode: 'train', 'val' or 'test'
    """
    if not cnf.augment_centers or mode != "train":
        return bbox
    assert isinstance(bbox, np.ndarray)
    rng = np.random.default_rng(seed=tid)
    o_x = rng.normal(loc=0, scale=cnf.augment_centers_std)
    o_y = rng.normal(loc=0, scale=cnf.augment_centers_std)
    bbox[:4] += np.array([o_x, o_y, o_x, o_y])
    return bbox


def compute_iou(
    bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, mode="iou", to_numpy=True
) -> torch.Tensor:
    result = {
        "iou": ops.box_iou,
        "giou": ops.generalized_box_iou,
        "diou": ops.distance_box_iou,
        "ciou": ops.complete_box_iou,
    }[mode](bboxes_a, bboxes_b)
    return result.numpy() if to_numpy else result


def relative_dist_size(bboxes_a: np.ndarray, bboxes_b: np.ndarray):
    # 2(xj − xi)/(hi + hj), 2(yj − yi)/(hi + hj), log (hi/hj), log (wi/wj)
    # bboxes_a: (n, 4) (format: ltbr)
    # bboxes_b: (m, 4) (format: ltbr)
    # return: (n, m, 4)
    n, m = bboxes_a.shape[0], bboxes_b.shape[0]
    bboxes_a = np.tile(bboxes_a[:, None, :], (1, m, 1))
    bboxes_b = np.tile(bboxes_b[None, :, :], (n, 1, 1))
    h_i = bboxes_a[:, :, 3] - bboxes_a[:, :, 1]
    h_j = bboxes_b[:, :, 3] - bboxes_b[:, :, 1]
    w_i = bboxes_a[:, :, 2] - bboxes_a[:, :, 0]
    w_j = bboxes_b[:, :, 2] - bboxes_b[:, :, 0]
    assert (
        np.all(h_i > 0) and np.all(h_j > 0) and np.all(w_i > 0) and np.all(w_j > 0)
    ), f"h_i: {h_i}, h_j: {h_j}, w_i: {w_i}, w_j: {w_j}. All should be > 0. (please use ltbr format)"
    return np.stack(
        [
            2 * (bboxes_b[:, :, 0] - bboxes_a[:, :, 0]) / (h_i + h_j),
            2 * (bboxes_b[:, :, 1] - bboxes_a[:, :, 1]) / (h_i + h_j),
            np.log(h_i / h_j),
            np.log(w_i / w_j),
        ],
        axis=2,
    )


def compute_normalized_l1(bboxes_a, bboxes_b):
    """Compute normalized L1 distance between two bounding boxes.
    Args:
        bboxes_a (Tensor): shape (n, 4)
        bboxes_b (Tensor): shape (m, 4)
    Returns:
        Tensor: shape (n, m)
    """

    # normalize bboxes
    for a in bboxes_a:
        for b in bboxes_b:
            X = a[0], a[2], b[0], b[2]
            Y = a[1], a[3], b[1], b[3]
            x_min, x_max, y_min, y_max = min(X), max(X), min(Y), max(Y)
            a[0], a[2] = (
                (a[0] - x_min) / (x_max - x_min),
                (a[2] - x_min) / (x_max - x_min),
            )
            a[1], a[3] = (
                (a[1] - y_min) / (y_max - y_min),
                (a[3] - y_min) / (y_max - y_min),
            )
            b[0], b[2] = (
                (b[0] - x_min) / (x_max - x_min),
                (b[2] - x_min) / (x_max - x_min),
            )
            b[1], b[3] = (
                (b[1] - y_min) / (y_max - y_min),
                (b[3] - y_min) / (y_max - y_min),
            )


def compute_normalized_l1_np(
    bboxes_a: np.ndarray, bboxes_b: np.ndarray, return_components: bool = False
) -> np.ndarray:
    """Compute normalized L1 distance between two bounding boxes.
    Args:
        bboxes_a (np.ndarray): shape (n, 4)
        bboxes_b (np.ndarray): shape (m, 4)
    Returns:
        np.ndarray: shape (n, m) if return_components is False
        np.ndarray: shape (n, m, 2) if return_components is True
    """
    n, m = bboxes_a.shape[0], bboxes_b.shape[0]
    dists = np.zeros((n, m)) if not return_components else np.zeros((n, m, 4))

    # normalize bboxes
    for row, a in enumerate(bboxes_a):
        for col, b in enumerate(bboxes_b):
            X = a[0], a[2], b[0], b[2]
            Y = a[1], a[3], b[1], b[3]
            x_min, x_max, y_min, y_max = min(X), max(X), min(Y), max(Y)
            X_A = (a[0] - x_min) / (x_max - x_min), (a[2] - x_min) / (x_max - x_min)
            Y_A = (a[1] - y_min) / (y_max - y_min), (a[3] - y_min) / (y_max - y_min)
            X_B = (b[0] - x_min) / (x_max - x_min), (b[2] - x_min) / (x_max - x_min)
            Y_B = (b[1] - y_min) / (y_max - y_min), (b[3] - y_min) / (y_max - y_min)

            if return_components:
                dists[row, col, :2] = np.abs(np.array(X_A) - np.array(X_B))
                dists[row, col, 2:] = np.abs(np.array(Y_A) - np.array(Y_B))
            else:
                dists[row, col] = (
                    np.abs(np.array(X_A) - np.array(X_B)).sum()
                    + np.abs(np.array(Y_A) - np.array(Y_B)).sum()
                )
    return dists


def compute_normalized_l1_np_vectorized(
    bboxes_a: np.ndarray, bboxes_b: np.ndarray, return_components: bool = False
) -> np.ndarray:
    """Compute normalized L1 distance between two bounding boxes.
    Args:
        bboxes_a (np.ndarray): shape (n, 4)
        bboxes_b (np.ndarray): shape (m, 4)
    Returns:
        np.ndarray: shape (n, m) if return_components is False
        np.ndarray: shape (n, m, 2) if return_components is True
    """
    n, m = bboxes_a.shape[0], bboxes_b.shape[0]
    dists = np.zeros((n, m, 4)) if return_components else np.zeros((n, m))

    matrix_bboxes_a_b = np.zeros((n, m, 8))
    matrix_bboxes_a_b[:, :, :4] = bboxes_a[:, np.newaxis, :]
    matrix_bboxes_a_b[:, :, 4:] = bboxes_b[np.newaxis, :, :]

    X = (
        matrix_bboxes_a_b[:, :, 0],
        matrix_bboxes_a_b[:, :, 2],
        matrix_bboxes_a_b[:, :, 4],
        matrix_bboxes_a_b[:, :, 6],
    )
    Y = (
        matrix_bboxes_a_b[:, :, 1],
        matrix_bboxes_a_b[:, :, 3],
        matrix_bboxes_a_b[:, :, 5],
        matrix_bboxes_a_b[:, :, 7],
    )
    x_min, x_max, y_min, y_max = (
        np.min(X, axis=0),
        np.max(X, axis=0),
        np.min(Y, axis=0),
        np.max(Y, axis=0),
    )

    X_A = (
        (matrix_bboxes_a_b[:, :, 0] - x_min) / (x_max - x_min),
        (matrix_bboxes_a_b[:, :, 2] - x_min) / (x_max - x_min),
    )
    Y_A = (
        (matrix_bboxes_a_b[:, :, 1] - y_min) / (y_max - y_min),
        (matrix_bboxes_a_b[:, :, 3] - y_min) / (y_max - y_min),
    )
    X_B = (
        (matrix_bboxes_a_b[:, :, 4] - x_min) / (x_max - x_min),
        (matrix_bboxes_a_b[:, :, 6] - x_min) / (x_max - x_min),
    )
    Y_B = (
        (matrix_bboxes_a_b[:, :, 5] - y_min) / (y_max - y_min),
        (matrix_bboxes_a_b[:, :, 7] - y_min) / (y_max - y_min),
    )

    if return_components:
        dists[:, :, :2] = np.abs(np.array(X_A) - np.array(X_B)).transpose(1, 2, 0)
        dists[:, :, 2:] = np.abs(np.array(Y_A) - np.array(Y_B)).transpose(1, 2, 0)
    else:
        dists = np.abs(np.array(X_A) - np.array(X_B)).sum(axis=0) + np.abs(
            np.array(Y_A) - np.array(Y_B)
        ).sum(axis=0)
    return dists


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_naive_features(img, x1, y1, x2, y2):
    """Get naive features for a bounding box.
    1. normalized width
    2. normalized height
    3. normalized aspect ratio
    4. normalized center x
    5. normalized center y

    Args:
        img (np.ndarray|torch.Tensor): shape (..., H, W)
        x1 (int): x1
        y1 (int): y1
        x2 (int): x2
        y2 (int): y2

    Returns:
        np.ndarray: shape (5,)
    """
    H, W = img.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return np.array(
        [
            (x2 - x1) / W,
            (y2 - y1) / H,
            (x2 - x1) / (y2 - y1),
            cx / W,
            cy / H,
        ]
    )


def get_naive_features_batch(img, bbs):
    """Get naive features for a batch of bounding boxes.
    1. normalized width
    2. normalized height
    3. normalized aspect ratio
    4. normalized center x
    5. normalized center y

    Args:
        img (np.ndarray|torch.Tensor): shape (..., H, W)
        bbs (np.ndarray): shape (N, 4)

    Returns:
        np.ndarray: shape (N, 5)
    """

    H, W = img.shape[:2]

    x1, y1, x2, y2 = bbs[:, 0], bbs[:, 1], bbs[:, 2], bbs[:, 3]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return np.array(
        [
            (x2 - x1) / W,
            (y2 - y1) / H,
            (x2 - x1) / (y2 - y1),
            cx / W,
            cy / H,
        ]
    ).T


def xywh2xyxy(bboxes):
    """Convert bounding boxes from xywh to xyxy format.
    Args:
        bboxes (np.ndarray): shape (N, 4)
    Returns:
        np.ndarray: shape (N, 4)
    """
    bboxes = bboxes.copy()
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes


def xyxy2xywh(bboxes):
    """Convert bounding boxes from xyxy to xywh format.
    Args:
        bboxes (np.ndarray): shape (N, 4)
    Returns:
        np.ndarray: shape (N, 4)
    """
    bboxes = bboxes.copy()
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    return bboxes
