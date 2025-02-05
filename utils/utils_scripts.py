import argparse
from enum import Enum
from functools import wraps
from pathlib import Path
from time import time
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn
from sklearn.metrics import average_precision_score

import wandb

Number = Union[int, float]
BBox = Sequence[Number]
CRBBox = Sequence[Number]

FOCAL = 1158
C_X = 960
C_Y = 540


class colors(Enum):
    FLAT_RED = (231, 76, 60)
    FLAT_GREEN = (46, 204, 113)
    FLAT_BLUE = (52, 152, 219)
    FLAT_PURPLE = (155, 89, 182)
    FLAT_ORANGE = (230, 126, 34)
    FLAT_YELLOW = (241, 196, 15)
    FLAT_BLACK = (44, 62, 80)
    FLAT_GREY = (149, 165, 166)


def to_color(color, colorspace="RGB"):
    """
    Returns a color tuple of 3 integers in range [0, 255].
    :param color: it can be an RGB color sequence of an HEX string
    :param colorspace: colorspace of the output tuple; it can be 'RGB' or 'BGR
    :return: tuple of 3 integers in range [0, 255] representing the input color
    """
    # hex color
    if isinstance(color, str) and color.startswith("#"):
        rgb_color = PIL.ImageColor.getcolor(color, "RGB")
        if colorspace == "BGR":
            rgb_color = rgb_color[::-1]
    else:
        rgb_color = [int(round(c)) for c in color]
    return tuple(rgb_color)


def is_list_empty(inList):
    if isinstance(inList, list):  # Is a list
        return all(map(is_list_empty, inList))
    if isinstance(inList, torch.Tensor):
        return inList.numel() == 0
    return False  # Not a list


def get_local_maxima(hmap, threshold, device="cuda"):
    """
    Find local maxima coordinates on the input heatmap `hmap`
    :param hmap: heatmap where you want to find the local maxima
    :param threshold: minimum value for a valid maximum
    :param device: device on which operations are made; values in {'cpu', 'cuda', 'cuda:<number>'}
    :return: tensor of N detected local maxima; shape (N, 2);
        >> each max is in the form (row_index, col_index)
    """

    if not isinstance(hmap, torch.Tensor):
        hmap = torch.from_numpy(hmap).float()

    hmap = hmap.to(device)

    map_left = torch.zeros(hmap.shape).to(device).float()
    map_left[1:, :] = hmap[:-1, :].clone()
    map_right = torch.zeros(hmap.shape).to(device).float()
    map_right[:-1, :] = hmap[1:, :].clone()
    map_up = torch.zeros(hmap.shape).to(device).float()
    map_up[:, 1:] = hmap[:, :-1].clone()
    map_down = torch.zeros(hmap.shape).to(device).float()
    map_down[:, :-1] = hmap[:, 1:].clone()

    p = torch.zeros(hmap.shape).to(device).float()
    p[hmap >= map_left] = 1
    p[hmap >= map_right] += 1
    p[hmap >= map_up] += 1
    p[hmap >= map_down] += 1
    p[hmap >= threshold] += 1

    p[p != 5] = 0

    return torch.nonzero(p).cpu()


def int_scale(x, scale_factor):
    """
    Apply scale factor to x and return the result as a rounded int
    :param x: number on which you want to apply the scale factor
    :param scale_factor: scale factor you want to apply to `x`
    :return: rounded result after the application of `scale_factor` to `x`
    """
    return int(round(x * scale_factor))


def map_over_img(overlay_map, rgb_img):
    """
    Apply the JET colormap to the input map and overlay it (with transparency 0.7) on `rgb_img` image.
    :param overlay_map: map you want to overlay on the input image; shape: (H_map, W_map)
    :param rgb_img: image you want to overlay the map on; shape: (H_img, W_img, 3)
    :return: result of the overlay (`overlay_map` over `rgb_img`)
        >> shape: (H_img, W_img, 3)
    """
    jet_hmap = cv2.applyColorMap(
        (overlay_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )[:, :, ::-1]
    jet_hmap = cv2.resize(jet_hmap, (rgb_img.shape[1], rgb_img.shape[0]))
    x = 0.7 * jet_hmap.astype(np.float) + 0.3 * rgb_img.astype(np.float)
    return x.astype(np.uint8)


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(y, mean, std, intround=False):
    x = y * std + mean
    return int(round(x)) if intround else x


def fagiano(a, b):
    max_value = max(a, b)
    return max_value / a, max_value / b


def create_mask(mask_h, mask_w, bboxes, scale_factor):
    mask = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        mask = cv2.rectangle(
            img=mask,
            color=(1, 1, 1),
            thickness=-1,
            pt1=(
                int_scale(x_min, scale_factor=scale_factor),
                int_scale(y_min, scale_factor=scale_factor),
            ),
            pt2=(
                int_scale(x_max, scale_factor=scale_factor),
                int_scale(y_max, scale_factor=scale_factor),
            ),
        )
    return mask


def torch2np(torch_tensor):
    """
    :param torch_tensor: input torch tensor
    :return: numpy array version of the input tensor
    """
    return torch_tensor.squeeze().cpu().numpy()


def torch2image(torch_image, vmin=0.0, vmax=1.0):
    """
    Convert input torch image to a an OpenCV-like numpy image
    :param torch_image: input torch image with shape (C, H, W) or (1, C, H, W) and values in range [vmin, vmax]
    :param vmin: minimum value for `torch_image`
    :param vmax: maximum value for `torch_image`
    :return: numpy image -> array with shape (H, W, C) with values in range [0, 255]
    """
    torch_image = (torch_image - vmin) / (vmax - vmin)
    return (torch2np(torch_image) * 255).astype(np.uint8).transpose((1, 2, 0))


def fill_missing_args(args: argparse.Namespace, parse_args_fn: callable):
    # fill in the missing arguments with the default values.
    # This is useful when you want to load a model that was trained with an older configuration
    default_args = parse_args_fn(default=True)
    for k, v in vars(default_args).items():
        if k not in args:
            args.__dict__[k] = v
    return args


def dclip(d, cnf, mode):
    if mode != "train":
        return d
    return float(np.clip(d, a_min=0, a_max=cnf.ds_stats["d_95_percentile"]))


def get_center(x: list) -> tuple:
    """Returns the center of a bounding box"""
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    return ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)


def imread(
    img_path: str, colorspace: str = "RGB", pil: bool = False
) -> Union[np.ndarray, Image.Image]:
    """
    Read image from path `img_path`
    :param img_path: path of the image to read
    :param colorspace: colorspace of the output image; must be one of {'RGB', 'BGR'}
    :param pil: if `True`, return a PIL.Image object, otherwise return a numpy array
    :return: PIL.Image object or numpy array in the specified colorspace
    """
    if pil:
        with open(img_path, "rb") as f:
            with PIL.Image.open(f) as img:
                return img.convert("RGB")
    else:
        img = cv2.imread(img_path)
        assert img is not None, f"'{img_path}' is not a valid image path"
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if colorspace == "RGB" else img


def init_wandb(
    args: argparse.Namespace, project_name: str, model_to_watch: nn.Module, resume: bool
):
    run_id = None
    resume = (
        resume and Path(args.exp_log_path / "run.id").exists()
    )  # resume only if the log file exists
    tmp_path = "/tmp"
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    if resume:
        with open(args.exp_log_path / "run.id", "r") as f:
            run_id = f.readline().strip()

    run = wandb.init(
        project=project_name,
        resume=resume,
        id=run_id,
        name=args.exp_name,
        dir=tmp_path,
        config=vars(args),
        entity="aimagelab-tracking",
        tags=[args.wandb_tag] if args.wandb_tag else None,
    )
    wandb.watch(model_to_watch)

    Path(args.exp_log_path).mkdir(parents=True, exist_ok=True)
    Path(args.exp_log_path / "run.id").write_text(str(run.id))


def log_on_wandb(
    metrics,
    y_true,
    y_pred,
    epoch,
    best_recall,
    best_precision,
    best_f1,
    best_th,
    ds_name,
):
    """
    Log metrics on wandb
    :param metrics: dict of metrics
    :param y_true: true labels (numpy array of 0 and 1 of shape (n_samples,))
    :param y_pred: predicted labels (numpy array of shape (n_samples,))
    """
    ds_name = f"{ds_name}_" if ds_name else ""
    metrics_wandb = {
        f"{ds_name}test/{k}": v for k, v in metrics.items() if not k.startswith("best_")
    }
    ap = average_precision_score(y_true, y_pred)

    metrics_wandb.update(
        {
            f"{ds_name}test_threshold/th_of_best_f1": best_th,
            f"{ds_name}test_threshold/recall_of_best_f1": best_recall,
            f"{ds_name}test_threshold/precision_of_best_f1": best_precision,
            f"{ds_name}test_threshold/best_f1": best_f1,
            f"{ds_name}test/AP": ap,
            "epoch": epoch,
        }
    )
    wandb.log(metrics_wandb)


def compute_at_threshold(y_true, y_pred, th):
    y_pred = y_pred > th
    # true positives, false positives, false negatives, true negatives
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return precision, recall, f1


def is_best_metric(result_metrics, best_metrics, best_metric_name):
    if best_metrics is None:
        return True
    compare_fn = (
        (lambda x, y: x < y)
        if best_metric_name == "test_loss"
        else (lambda x, y: x > y)
    )
    return bool(
        compare_fn(result_metrics[best_metric_name], best_metrics[best_metric_name])
    )


def to3d(x2d, y2d, cam_dist, fx, fy, cx, cy):
    """
    Converts a 2D point on the image plane into a 3D point in the standard
    coordinate system using the intrinsic camera parameters.

    :param x2d: 2D image coordinate [px]
    :param y2d: 2D y coordinate [px]
    :param cam_dist: distance from camera [m]
    :param fx: image component of the focal length
    :param fy: y component of the focal length
    :param cx: image component of the central point
    :param cy: y component of the central point
    :return: 3D coordinates
    """

    k = (-1) * np.sqrt(
        fx**2 * fy**2
        + fx**2 * cy**2
        - 2 * fx**2 * cy * y2d
        + fx**2 * y2d**2
        + fy**2 * cx**2
        - 2 * fy**2 * cx * x2d
        + fy**2 * x2d**2
    )

    x3d = ((fy * cam_dist * cx) - (fy * cam_dist * x2d)) / k
    y3d = ((fx * cy * cam_dist) - (fx * cam_dist * y2d)) / k
    z3d = -(fx * fy * cam_dist) / k

    return x3d, y3d, z3d


def to3d_numpy(xy2d, cam_dist, fx, fy, cx, cy):
    """
    Converts a 2D point on the image plane into a 3D point in the standard
    coordinate system using the intrinsic camera parameters.

    :param xy2d: 2D image coordinates [px]
    :param cam_dist: distance from camera [m]
    :param fx: image component of the focal length
    :param fy: y component of the focal length
    :param cx: image component of the central point
    :param cy: y component of the central point
    :return: 3D coordinates
    """
    x = xy2d[:, 0:1]
    y = xy2d[:, 1:2]

    k = (-1) * np.sqrt(
        fx**2 * fy**2
        + fx**2 * cy**2
        - 2 * fx**2 * cy * y
        + fx**2 * y**2
        + fy**2 * cx**2
        - 2 * fy**2 * cx * x
        + fy**2 * x**2
    )

    x3d = ((fy * cam_dist * cx) - (fy * cam_dist * x)) / k
    y3d = ((fx * cy * cam_dist) - (fx * cam_dist * y)) / k
    z3d = -(fx * fy * cam_dist) / k

    return np.concatenate([x3d, y3d, z3d], axis=1)


def to2d(
    points_nx3: List[np.ndarray], fx: float, fy: float, cx: float, cy: float
) -> Union[Any, np.ndarray]:
    """
    Converts a 3D point on the image plane into a 2D point in the standard
    coordinate system using the intrinsic camera parameters.

    :param points_nx3: 3D [(x,y,z),...] coordinates
    :param fx: x component of the focal length
    :param fy: y component of the focal length
    :param cx: x component of the central point
    :param cy: y component of the central point
    :return: 2D coordinates
    """

    points_2d, _ = cv2.projectPoints(
        np.array(points_nx3),
        (0, 0, 0),
        (0, 0, 0),
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
        np.array([]),
    )

    return np.column_stack(
        [np.squeeze(points_2d, axis=1), np.array(points_nx3).transpose(1, 0)[2]]
    )


def from2dot5dto3d(
    my2dot5coords: torch.Tensor,
    scale: Tuple[float] = 1,
    f: Union[Tuple[int, int], int] = FOCAL,
    c: Tuple[int, int] = (C_X, C_Y),
) -> torch.Tensor:
    if isinstance(f, int):
        f = (f, f)

    x2d, y2d, cam_dist = (
        my2dot5coords[:, 0] * scale[0],
        my2dot5coords[:, 1] * scale[1],
        my2dot5coords[:, 2],
    )
    fx, fy = f
    cx, cy = c

    k = (-1) * torch.sqrt(
        fx**2 * fy**2
        + fx**2 * cy**2
        - 2 * fx**2 * cy * y2d
        + fx**2 * y2d**2
        + fy**2 * cx**2
        - 2 * fy**2 * cx * x2d
        + fy**2 * x2d**2
    )

    x3d = ((fy * cam_dist * cx) - (fy * cam_dist * x2d)) / k
    y3d = ((fx * cy * cam_dist) - (fx * cam_dist * y2d)) / k
    z3d = -(fx * fy * cam_dist) / k

    return torch.cat([x3d.unsqueeze(1), y3d.unsqueeze(1), z3d.unsqueeze(1)], dim=1)


EPS = torch.finfo(torch.float32).eps
NP_EPS = np.finfo(np.float32).eps


def z_to_nearness(
    x: Union[torch.Tensor, np.ndarray, list],
) -> Union[torch.Tensor, np.ndarray, list]:
    if isinstance(x, np.ndarray):
        return -np.log(x + NP_EPS)
    elif isinstance(x, torch.Tensor):
        return -torch.log(x + EPS)
    elif isinstance(x, list):
        return (-np.log(np.array(x) + NP_EPS)).tolist()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def nearness_to_z(
    x: Union[torch.Tensor, np.ndarray, list],
) -> Union[torch.Tensor, np.ndarray, list]:
    if isinstance(x, np.ndarray):
        return np.exp(-x)
    elif isinstance(x, torch.Tensor):
        return torch.exp(-x)
    elif isinstance(x, list):
        return np.exp(-np.array(x)).tolist()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func: %r, took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


class Timer:
    def __init__(self, string=None, one_line=False):
        self.string = string
        self.one_line = one_line
        self.end = "" if one_line else "\n"

    def __enter__(self):
        print(self.string, end=self.end)
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start
        unit = "s"
        if self.interval < 1:
            self.interval *= 1000
            unit = "ms"
        if self.one_line:
            print(f" in: {self.interval:3.3f} {unit}")
        else:
            print(f"Elapsed time: {self.interval:3.3f} {unit}\n")


if __name__ == "__main__":
    print(fagiano(0.37 / 100, 0.05 / 1.0))
