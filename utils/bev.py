import argparse
import math
from pathlib import Path
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F
from colour import Color
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle

camera_params = {
    "motsynth": [
        [1158, 0.00, 1280 / 2],  # MOTSYNTH CAMERA MATRIX
        [0.00, 1158, 720 / 2],
        [0.00, 0.00, 1.0],
    ],
    "kitti": [
        [721.5377, 0.00, 621],  # KITTI CAMERA MATRIX
        [0.00, 721.5377, 187.5],
        [0.00, 0.00, 1.0],
    ],
    "nuscenes": [
        [1266.42, 0.0, 783.73],  # NUSCENES CAMERA MATRIX
        [0.0, 1266.42, 491.50],
        [0.0, 0.0, 1.0],
    ],
}
visual_features = {
    "motsynth": {
        "colorbar_error": dict(
            orientation="vertical", pad=0.01, shrink=0.388, aspect=30, fraction=0.025
        ),
        "circle_radius": 1.3,
        "offset_circle_text_h": 0,
        "offset_circle_text_v": 0,
        "max_dist": 50,
    },
    "kitti": {
        "colorbar_error": dict(
            orientation="vertical", pad=0.01, shrink=0.26, aspect=19, fraction=0.025
        ),
        "circle_radius": 1.3,
        "offset_circle_text_h": 0.9,
        "offset_circle_text_v": 0.2,
        "max_dist": 50,
    },
    "nuscenes": {
        "colorbar_error": dict(
            orientation="vertical",
            shrink=0.36,
            aspect=20,
            fraction=0.15,
            pad=0.05,
        ),
        "circle_radius": 1.3,
        "offset_circle_text_h": 0.9,
        "offset_circle_text_v": 0.2,
        "max_dist": 50,
    },
}
figure_ratios = {
    "motsynth": {"width_ratios": [1, 0.65]},
    "kitti": {"width_ratios": [1.15, 1]},
    "nuscenes": {"width_ratios": [1.15, 1]},
}

assert (
    camera_params.keys() == visual_features.keys()
), "Camera parameters and visual features dataset mismatch"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch_elem", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, default="motsynth", choices=camera_params.keys()
    )
    parser.add_argument(
        "--no_numbers",
        action="store_true",
        help="Do not display distances on the bounding boxes",
    )
    return parser.parse_args()


def extract_output(dataset, output):
    if dataset == "kitti":
        ck = output
    elif dataset == "motsynth":
        ck = output[0]
    return ck.detach().cpu().numpy().astype(np.float32)


def show_bev(
    dataset,
    dists_pred,
    dists_vars,
    video_bboxes,
    clip_clean,
    y_true,
    out_path,
    video,
    frame,
):
    for batch in range(len(video_bboxes)):
        batch_lens = np.insert(np.cumsum([len(v[0]) for v in video_bboxes]), 0, 0)
        bboxes = video_bboxes[batch][-1].detach().cpu().numpy().astype(np.float32)
        y_true_elem = (
            y_true[batch_lens[batch] : batch_lens[batch + 1]]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        dists_pred_elem = dists_pred[batch_lens[batch] : batch_lens[batch + 1]]
        dists_vars_elem = None
        if dataset == "motsynth":
            dists_vars_elem = dists_vars[batch_lens[batch] : batch_lens[batch + 1]]
        clip_clean_elem = (
            clip_clean[batch, :3, 0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        assert dists_pred_elem.shape == y_true_elem.shape

        if dists_pred_elem.shape[0] == 0:
            continue

        fig, ax = compute_bev(
            dataset,
            dists_pred_elem,
            dists_vars_elem,
            bboxes,
            clip_clean_elem,
            y_true_elem,
        )
        Path(out_path).mkdir(exist_ok=True)
        fig.savefig(
            (Path(out_path) / f"bev_{batch}.png").with_suffix(".png"),
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )


def compute_bev(dataset, dists_pred, dists_vars, bboxes, clip_clean, y_true):
    # display bounding boxes on the image and BEV map bottom
    fig, ax = plt.subplots(
        1, 2, figsize=(12, 8), gridspec_kw=figure_ratios[dataset]
    )  # 6 8
    fig.set_tight_layout(True)
    ax[0].imshow(clip_clean)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_axis_off()  # remove axis

    bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, :2]
    # display bounding boxes on the image ordered by distance
    ordered_idx = np.argsort(dists_pred)[::-1]
    for idx in ordered_idx:
        bbox = bboxes[idx]
        d = np.clip(dists_pred[idx], 0, int(dists_pred.max()))
        color = "tab:blue"
        ax[0].add_patch(
            Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        t = ax[0].text(
            bbox[0] - 1, bbox[1] - 8, f"{d:.1f}", color="white", fontdict={"size": 8}
        )
        t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color, pad=0.5))

    cx_cy = bboxes[:, :2] + bboxes[:, 2:] / 2
    xyz = pixel_to_camera(cx_cy, camera_params[dataset], 1).numpy()
    xyz_gt = xyz.copy()
    for i, bbox in enumerate(bboxes):
        d = np.clip(dists_pred[i], 0, int(dists_pred.max()))
        xyz[i, :] *= d
        xyz_gt[i, :] *= y_true[i]

    z_max = 40 if dataset == "motsynth" else xyz[:, 2].max()
    error_multiplier = 10
    error_max = max(
        math.ceil(
            (
                round(np.linalg.norm(xyz - xyz_gt, axis=1).max() * error_multiplier)
                / error_multiplier
            )
            * 1.2
        ),
        10,
    )
    error_colors = [
        color.hex
        for color in Color("green").range_to(
            Color("red"), int(round(error_max * error_multiplier)) + 1
        )
    ]
    for i, bbox in enumerate(bboxes):
        d = np.clip(dists_pred[i], 0, int(dists_pred.max()))
        error = np.linalg.norm(xyz[i] - xyz_gt[i])
        color = error_colors[int(round(error * error_multiplier))]
        # add circles to the BEV map
        radius = visual_features[dataset]["circle_radius"] * (
            z_max / visual_features[dataset]["max_dist"]
        )
        ax[1].add_patch(
            Circle((xyz[i, 0], xyz[i, 2]), radius=radius, fill=True, color=color)
        )
        ax[1].scatter(xyz_gt[i, 0], xyz_gt[i, 2], marker="x", color="black", s=20)

    uv_max = [0.0, clip_clean.shape[0]]
    xyz_max = pixel_to_camera(uv_max, camera_params[dataset], z_max)
    ax[1].set_ylim(0, xyz[:, 2].max() + 1)
    x_max = abs(xyz_max[0])
    z_max_plot = np.ceil(z_max / 10) * 10 + 1
    ax[1].plot(
        [0, x_max], [0, z_max_plot], "--", color="black", linewidth=2, alpha=0.75
    )
    ax[1].plot(
        [0, -x_max], [0, z_max_plot], "--", color="black", linewidth=2, alpha=0.75
    )
    ax[1].set_xlim(-x_max, x_max)
    ax[1].set_ylim(0, z_max_plot)
    ax[1].set_aspect("equal")
    if dataset == "kitti":
        ax[1].set_xlabel(r"$\mathrm{x}$ $\mathrm{[m]}$")
    ax[1].set_ylabel(r"Distance $\mathrm{[m]}$")
    ax[1].spines[["left", "right", "top"]].set_visible(False)
    ax[1].grid(True, axis="y", color="gray", linestyle="--", linewidth=0.5)
    ax[1].tick_params(axis="both", which="both", length=0)

    # add colorbar to ax[1] (BEV) measuring error
    norm = mpl.colors.Normalize(vmin=0, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap("RdYlGn_r"))
    cmap.set_array([])
    # cax = fig.add_axes([ax[1].get_position().x1+0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    cbar = plt.colorbar(
        cmap, ax=ax[1], **visual_features[dataset]["colorbar_error"], location="right"
    )
    cbar.set_label(
        r"Error $\mathrm{[m]}$",
    )
    cbar.set_ticks(np.arange(0, error_max + 0.1, 1))
    cbar.set_ticklabels(np.arange(0, error_max + 1, 1))

    artist = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            color="black",
            linewidth=0,
            alpha=1,
            marker="o",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            color="black",
            linewidth=0,
            alpha=1,
            marker="x",
            markersize=8,
        ),
    ]
    ax[1].legend(
        artist,
        ["Prediction", "Ground Truth"],
        loc="upper right",
        fontsize=10,
        framealpha=1,
        handlelength=1,
        fancybox=False,
        edgecolor="black",
    )

    return fig, ax


def pixel_to_camera(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
    where x is the number of keypoints
    """
    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(
            0, 2, 1
        )  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(
        uv_tensor, pad=(0, 1), mode="constant", value=1
    )  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)
    return torch.matmul(uv_padded, kk_1.t()) * z_met


def main():
    args = get_args()
    show_bev(args)


if __name__ == "__main__":
    main()
