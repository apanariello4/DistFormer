from typing import Optional

import numpy as np


def abs_rel_diff(y_true: np.ndarray, y_pred: np.ndarray):
    """Absolute relative difference."""
    return np.mean(np.abs(y_pred - y_true) / (y_true))


def squa_rel_diff(y_true: np.ndarray, y_pred: np.ndarray):
    """Squared relative difference."""
    return np.mean(np.square(y_pred - y_true) / (y_true))


def rmse_linear(y_true: np.ndarray, y_pred: np.ndarray):
    """Root mean squared error."""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray):
    """Root mean squared error."""
    EPS = np.finfo(y_true.dtype).eps
    y_pred = np.clip(y_pred, EPS, None)
    return np.sqrt(np.mean(np.square(np.log(y_pred) - np.log(y_true + EPS))))


def threshold_accuracy(y_true: np.ndarray, y_pred: np.ndarray, th: float = 1.25):
    """Threshold accuracy."""
    threshold = np.maximum((y_true / y_pred), (y_pred / y_true))
    th_accuracy = np.mean(threshold < th)
    return th_accuracy if np.isfinite(th_accuracy) else 0


def rel_dist_error(y_true: np.ndarray, y_pred: np.ndarray, th: float = 0.05):
    # calculate relative distance errors for each object and then percentage below threshold
    if len(y_true) == 0:
        return 0
    dist_errors = abs((y_true - y_pred) / y_true)

    return float(len(dist_errors[dist_errors < th])) / len(y_true) * 100


def ale(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Average localization error."""
    th = [0, 10, 20, 30, 100]
    dist = np.abs(y_true - y_pred)
    ale_ = {
        f"ale_{th1}-{th2}": np.mean(dist[(y_true >= th1) & (y_true < th2)])
        for th1, th2 in zip(th[:-1], th[1:])
    }
    ale_all = {"ale_all": np.mean(dist)}
    if (y_true > th[-1]).any():
        ale_ |= {"ale_100+": np.mean(dist[y_true >= 100])}
    return ale_ | ale_all


def alp(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Average localization precision."""
    th = [0.5, 1, 2]
    dist = np.abs(y_true - y_pred)
    return {
        f"alp_@{th1}m": (len(dist[dist < th1]) / len(dist)) if len(dist) != 0 else None
        for th1 in th
    }


def aloe(y_true: np.ndarray, y_pred: np.ndarray, y_occlusions: np.ndarray) -> dict:
    """Average localization of occluded objects error."""
    th = [0.0, 0.3, 0.5, 0.75, 1.0]
    dist = np.abs(y_true - y_pred)
    aloe_ = {
        f"aloe_{th1}-{th2}": np.mean(dist[(y_occlusions >= th1) & (y_occlusions < th2)])
        for th1, th2 in zip(th[:-1], th[1:])
    }
    return aloe_


def aloe_kitti(
    y_true: np.ndarray, y_pred: np.ndarray, y_occlusions: np.ndarray
) -> dict:
    """Average localization of occluded objects error."""
    th = np.unique(y_occlusions)
    dist = np.abs(y_true - y_pred)
    aloe_ = {f"aloe_{th1}": np.mean(dist[(y_occlusions == th1)]) for th1 in th}
    aloe_all = {"aloe_all": np.mean(dist)}
    return aloe_ | aloe_all


def get_metrics_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, y_visibilities: Optional[list] = None
) -> dict:
    ale_ = ale(y_true, y_pred)
    alp_ = alp(y_true, y_pred)

    if (
        y_visibilities is not None
        and y_visibilities.shape[0] > 0
        and not np.all(y_visibilities == y_visibilities[0])
    ):
        if (y_visibilities > 1).any():
            aloe_ = aloe_kitti(y_true, y_pred, y_visibilities)
        else:
            y_occlusions = 1 - np.array(y_visibilities)
            aloe_ = aloe(y_true, y_pred, y_occlusions)
    else:
        aloe_ = {}

    return (
        {
            "abs_rel_diff": abs_rel_diff(y_true, y_pred),
            "squa_rel_diff": squa_rel_diff(y_true, y_pred),
            "rmse_linear": rmse_linear(y_true, y_pred),
            "rmse_log": rmse_log(y_true, y_pred),
            "delta_1": threshold_accuracy(y_true, y_pred),
            "delta_2": threshold_accuracy(y_true, y_pred, th=1.25**2),
            "delta_3": threshold_accuracy(y_true, y_pred, th=1.25**3),
            "rel_dist_5": rel_dist_error(y_true, y_pred, th=0.05),
            "rel_dist_10": rel_dist_error(y_true, y_pred, th=0.1),
            "rel_dist_15": rel_dist_error(y_true, y_pred, th=0.15),
        }
        | ale_
        | alp_
        | aloe_
    )


def process_values(
    y_true: list,
    y_pred: list,
    max_dist: float,
    y_visibilities: Optional[list] = None,
    y_classes: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_visibilities = np.array(y_visibilities) if y_visibilities is not None else None
    y_classes = np.array(y_classes) if y_classes is not None else None
    # filter out values where y_true is 0
    mask = y_true > 0
    if max_dist is not None and max_dist > 0:
        mask = mask & (y_true < max_dist)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_visibilities is not None:
        y_visibilities = y_visibilities[mask]
    if y_classes is not None:
        y_classes = y_classes[mask]
    return y_true, y_pred, y_visibilities, y_classes


def get_metrics(
    y_true: list,
    y_pred: list,
    y_classes: list,
    classes_mapping: dict,
    max_dist: float,
    y_visibilities: Optional[list] = None,
    long_range=True,
) -> dict:
    """classes_mapping: {class_name: class_id}"""

    y_true, y_pred, y_visibilities, y_classes = process_values(
        y_true, y_pred, max_dist, y_visibilities, y_classes
    )

    results = {"all": get_metrics_per_class(y_true, y_pred, y_visibilities)}

    id2name = {v: k for k, v in classes_mapping.items()}
    unique_classes = np.unique(y_classes)
    available_classes = {
        cls_id: cls_name
        for cls_id, cls_name in id2name.items()
        if cls_id in unique_classes
    }
    for cls_id, cls_name in available_classes.items():
        y_visibilities_cls = (
            y_visibilities[y_classes == cls_id] if y_visibilities is not None else None
        )
        results[cls_name] = get_metrics_per_class(
            y_true[y_classes == cls_id], y_pred[y_classes == cls_id], y_visibilities_cls
        )

    if long_range and "Car" in available_classes.values():
        y_visibilities_long_range = (
            np.array(y_visibilities)[y_true > 40]
            if y_visibilities is not None
            else None
        )
        y_classes_long_range = y_classes[y_true > 40]
        y_pred_long_range = y_pred[y_true > 40]
        y_true_long_range = y_true[y_true > 40]
        results["long_range"] = get_metrics_per_class(
            y_true_long_range[y_classes_long_range == cls_id],
            y_pred_long_range[y_classes_long_range == cls_id],
            y_visibilities_long_range[y_classes_long_range == cls_id],
        )

    return results


PURPLE = "\033[95m"
CYAN = "\033[96m"
DARKCYAN = "\033[36m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
END = "\033[0m"


def print_metrics(metrics: dict) -> None:
    print(f"{BOLD}Metrics")
    print(f"{'':<16}", ("{:<12}" * len(metrics)).format(*metrics.keys()) + END)
    for metric_name in metrics["all"]:
        all_classes_metric = [
            metrics[cls_id].get(metric_name, -1) for cls_id in metrics
        ]
        arrow = (
            f"{BOLD}{GREEN} ↑{END}"
            if metric_name.startswith(("delta", "alp", "rel"))
            else f"{BOLD}{RED} ↓{END}"
        )
        arrow = "" if metric_name.startswith("num") else arrow
        print(
            f"{metric_name:<13}{arrow} ",
            ("{:<12.4f}" * len(all_classes_metric)).format(*all_classes_metric),
        )
