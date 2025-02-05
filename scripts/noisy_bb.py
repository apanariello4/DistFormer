import numpy as np


def noisy_bb(
    bboxes,
    screen_h: int,
    screen_w: int,
    min_iou=0,
    min_offset=0.0,
    max_offset=0.01,
    min_scale=0.9,
    max_scale=1.1,
):
    original_bboxes = bboxes.copy()

    offset = (
        np.random.rand(bboxes.shape[0], 2) * (max_offset - min_offset) + min_offset) * np.array([screen_w, screen_h])
    scale = np.random.rand(bboxes.shape[0], 2) * (max_scale - min_scale) + min_scale

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + offset[:, 0:1]  # apply offset to x coords
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + offset[:, 1:2]  # apply offset to y coords

    x_diff = bboxes[:, 2] - bboxes[:, 0]  # get width of box
    y_diff = bboxes[:, 3] - bboxes[:, 1]  # get height of box
    scale = (1 - scale) / 2

    # scale zooms in/out from the center of the box
    bboxes[:, 0] = np.where(
        scale[:, 0] > 1,
        bboxes[:, 0] + (x_diff * scale[:, 0]),
        bboxes[:, 0] - (x_diff * scale[:, 0]),
    )
    bboxes[:, 1] = np.where(
        scale[:, 1] > 1,
        bboxes[:, 1] + (y_diff * scale[:, 1]),
        bboxes[:, 1] - (y_diff * scale[:, 1]),
    )
    bboxes[:, 2] = np.where(
        scale[:, 0] > 1,
        bboxes[:, 2] - (x_diff * scale[:, 0]),
        bboxes[:, 2] + (x_diff * scale[:, 0]),
    )
    bboxes[:, 3] = np.where(
        scale[:, 1] > 1,
        bboxes[:, 3] - (y_diff * scale[:, 1]),
        bboxes[:, 3] + (y_diff * scale[:, 1]),
    )

    iou_val = iou_pairs(bboxes, original_bboxes)
    assert (
        iou_val.min() > min_iou
    ), f"Found iou_val.min() = {iou_val.min()} < {min_iou} = min_iou"

    return bboxes


def noisy_bb_iou_preserving(bboxes, min_iou):
    # taken from: https://github.com/dvl-tum/mot_neural_solver/blob/4541eb605a922876c376ec95d4e55509175d145e/src/mot_neural_solver/data/augmentation.py#L41
    """
    Randomly perturbs bounding box coordinates by applying small shifts to each side of the box.
    Shifts are computed so that the resulting bounding box always has an IoU with he original one that is, at least
    dataset_params['min_iou_bb_wiggling']

    To do so, we make sure that the relative distortion of each side has the appropriate value. Let 1 > max_eps > 0 be
    the max 'distortion' of the new height and width, so that each side is modified by eps in (-max_eps, max_eps) as
    new_bb_top = old_bb_top + (1 + eps)*old_height (Analogous for the remaining sides).
    Hence old_height*(1-max_eps) <= new_height <= (1 + max_eps)*old_height, and analogously for width.
    Then, max distortion happens when the new box is completely contained in the old box, and IoU
    is (1-2*max_eps)^2. By imposing that this amount is greater or equal than dataset_params['min_iou_bb_wiggling'].
    we get our desired maximum epsilon, and we get a 'safe' range in which to sample epsilons.

    All coordinate based columns in graph_df are updated accordingly
    """
    original_bboxes = bboxes.copy()

    upper_bound_inside_box = (1 - np.sqrt(min_iou)) / 2
    upper_bound_outside_box = 0.5 * (1 / np.sqrt(min_iou) - 1)
    max_eps = min(upper_bound_inside_box, upper_bound_outside_box)

    epsilons = np.random.uniform(-max_eps, max_eps, 4 * bboxes.shape[0]).reshape(-1, 4)
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    coord_shift_vals = np.array([h, h, w, w], dtype=np.float32).T * epsilons
    bb_top_shift, bb_bot_shift, bb_left_shift, bb_right_shift = coord_shift_vals.T

    bboxes[:, 0] += bb_left_shift
    bboxes[:, 1] += bb_top_shift
    bboxes[:, 2] += bb_right_shift
    bboxes[:, 3] += bb_bot_shift

    # Just make sure that we did not surpass the IoU threshold (dataset_params['min_iou_bb_wiggling'])
    iou_val = iou_pairs(bboxes, original_bboxes)
    assert (
        iou_val.min() > min_iou
    ), f"Found iou_val.min() = {iou_val.min()} < {min_iou} = min_iou"


def iou_pairs(boxA, boxB):
    """
    Args:
        boxA: numpy array of bounding boxes with size (N, 4).
        boxB: numpy array of bounding boxes with size (N, 4)

    Returns:
        numpy array of size (N,), where the ith element is the IoU between the ith box in boxA and boxB.

    Note: bounding box coordinates are given in format (top, left, bottom, right)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:, 0], boxB[:, 0])
    yA = np.maximum(boxA[:, 1], boxB[:, 1])
    xB = np.minimum(boxA[:, 2], boxB[:, 2])
    yB = np.minimum(boxA[:, 3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    return interArea / (boxAArea + boxBArea - interArea).astype(float)
