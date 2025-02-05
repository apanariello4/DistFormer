import argparse

import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from colour import Color
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command
from matplotlib import rc

mpl.rcParams["text.usetex"] = True
rc("font", family="sans-serif")  # , sans-serif='Times')
mpl.rcParams.update(
    {
        "text.usetex": True,
        #     "font.family": "serif",
        #     "font.sans-serif": ["Times"]})
        # "font.family": "STIXGeneral",
        # "font.sans-serif": "Helvetica",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "mathtext.fontset": "stix",
    }
)
plt.rcParams.update({"font.size": 15})

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
}

visual_features = {
    "motsynth": {
        "colorbar_error": dict(
            orientation="vertical", pad=0.01, shrink=0.388, aspect=30, fraction=0.025
        ),
        "colorbar_dist": dict(
            orientation="vertical", pad=0.01, shrink=0.40, aspect=30, fraction=0.025
        ),
        "circle_radius": 1.3,
        "offset_circle_text_h": 0,
        "offset_circle_text_v": 0,
    },
    "kitti": {
        "colorbar_error": dict(
            orientation="vertical", pad=0.01, shrink=0.26, aspect=19, fraction=0.025
        ),
        "colorbar_dist": dict(
            orientation="vertical", pad=0.01, shrink=0.215, aspect=18, fraction=0.025
        ),
        "circle_radius": 2.75,
        "offset_circle_text_h": 0.9,
        "offset_circle_text_v": 0.2,
    },
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


def show_bev(args):
    ck = torch.load(args.ckpt)
    output = extract_output(args.dataset, ck["output"])  # output.shape == y_true.shape
    bboxes = (
        ck["video_bboxes"][args.batch_elem][0].detach().cpu().numpy().astype(np.float32)
    )
    clip_clean = (
        ck["clip_clean"][args.batch_elem, :3, 0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    y_true = ck["y_true"].detach().cpu().numpy().astype(np.float32)
    batch_lens = np.insert(np.cumsum([len(v[0]) for v in ck["video_bboxes"]]), 0, 0)
    output = output[batch_lens[args.batch_elem] : batch_lens[args.batch_elem + 1]]
    y_true = y_true[batch_lens[args.batch_elem] : batch_lens[args.batch_elem + 1]]
    assert output.shape == y_true.shape

    # display bounding boxes on the image and BEV map bottom
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), width_ratios=[1.35, 1])  # 6 8
    fig.set_tight_layout(True)
    ax[0].imshow(clip_clean)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_axis_off()  # remove axis

    bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, :2]

    # display bounding boxes on the image ordered by distance
    colors = [
        color.hex
        for color in Color("green").range_to(Color("red"), int(output.max()) + 1)
    ]
    ordered_idx = np.argsort(output)[::-1]
    for idx in ordered_idx:
        bbox = bboxes[idx]
        d = np.clip(output[idx], 0, int(output.max()))
        gt_d = np.clip(y_true[idx], 0, int(output.max()))
        color = colors[int(round(d))]
        ax[0].add_patch(
            Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        if not args.no_numbers:
            ax[0].text(
                bbox[0],
                bbox[1] - 5,
                f"{d:.1f}",
                color=color,
                fontdict={"weight": "bold", "size": 12},
            )

    cx_cy = bboxes[:, :2] + bboxes[:, 2:] / 2
    xyz = pixel_to_camera(cx_cy, camera_params[args.dataset], 1).numpy()
    xyz_gt = xyz.copy()
    for i, bbox in enumerate(bboxes):
        d = np.clip(output[i], 0, int(output.max()))
        xyz[i, :] *= d
        xyz_gt[i, :] *= y_true[i]

    z_max = xyz[:, 2].max()
    error_multiplier = 10
    error_max = 6
    error_colors = [
        color.hex
        for color in Color("green").range_to(
            Color("red"), int(round(error_max * error_multiplier)) + 1
        )
    ]
    for i, bbox in enumerate(bboxes):
        d = np.clip(output[i], 0, int(output.max()))
        # d_gt = np.clip(y_true[i], 0, int(output.max()))
        error = np.linalg.norm(xyz[i] - xyz_gt[i])
        color = error_colors[int(round(error * error_multiplier))]
        # add circles to the BEV map
        ax[1].add_patch(
            Circle(
                (xyz[i, 0], xyz[i, 2]),
                radius=visual_features[args.dataset]["circle_radius"],
                fill=True,
                color=color,
            )
        )
        # ax[1].add_patch(Circle((xyz_gt[i, 0], xyz_gt[i, 2]), radius=1, fill=False, color='black'))
        # add text to the BEV map
        # str_ = f'{error:.1f}'
        # x_offset = 0.3 * len(str_) + visual_features[args.dataset]['offset_circle_text_h']
        # ax[1].text(xyz[i, 0] - x_offset, xyz[i, 2] - 0.4 - visual_features[args.dataset]['offset_circle_text_v']
        #            , str_, color='black', fontsize=5)
        # ax[1].text(xyz_gt[i, 0], xyz_gt[i, 2] + 1, f'{d_gt:.1f}', color='black', fontsize=8)

    uv_max = [0.0, clip_clean.shape[0]]
    xyz_max = pixel_to_camera(uv_max, camera_params[args.dataset], z_max)
    ax[1].set_ylim(0, xyz[:, 2].max() + 1)
    x_max = abs(xyz_max[0])
    corr = round(float(x_max / 3))
    z_max_plot = np.ceil(z_max / 10) * 10 + 1
    ax[1].plot(
        [0, x_max], [0, z_max_plot], "--", color="black", linewidth=2, alpha=0.75
    )
    ax[1].plot(
        [0, -x_max], [0, z_max_plot], "--", color="black", linewidth=2, alpha=0.75
    )
    ax[1].set_xlim(-x_max - corr, x_max + corr)
    ax[1].set_ylim(0, z_max_plot)
    ax[1].set_aspect("equal")
    if args.dataset == "kitti":
        ax[1].set_xlabel(r"$\mathrm{x}$ $\mathrm{[m]}$")
    ax[1].set_ylabel(r"$\mathrm{z}$ $\mathrm{[m]}$")
    # ax[1].set_title('Bird\'s Eye View', fontsize=18, fontfamily='serif', fontweight='bold')
    ax[1].spines[["left", "right", "top"]].set_visible(False)
    # ax[1].grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5)
    ax[1].grid(True, axis="y", color="gray", linestyle="--", linewidth=0.5)
    ax[1].tick_params(axis="both", which="both", length=0)

    # add colorbar to ax[0] (image) measuring distance
    norm = mpl.colors.Normalize(vmin=0, vmax=int(output.max()))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap("RdYlGn_r"))
    cmap.set_array([])
    cbar = plt.colorbar(
        cmap,
        ax=ax[0],
        **visual_features[args.dataset]["colorbar_dist"],
        location="left",
    )
    cbar.set_label("Predicted distance $\mathrm{[m]}$")
    cbar.set_ticks(np.arange(0, int(output.max()) + 1, 10))
    cbar.set_ticklabels(np.arange(0, int(output.max()) + 1, 10))

    # add colorbar to ax[1] (BEV) measuring error
    norm = mpl.colors.Normalize(vmin=0, vmax=error_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap("RdYlGn_r"))
    cmap.set_array([])
    # cax = fig.add_axes([ax[1].get_position().x1+0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    cbar = plt.colorbar(
        cmap,
        ax=ax[1],
        **visual_features[args.dataset]["colorbar_error"],
        location="right",
    )
    cbar.set_label(
        r"Error $\mathrm{[m]}$",
    )
    cbar.set_ticks(np.arange(0, error_max + 1, 1))
    cbar.set_ticklabels(np.arange(0, error_max + 1, 1))

    # add to ax[1] a circle lower bottom to indicate that each Circle is a person, add a text "person" next to it
    # ax[1].add_patch(Circle((-x_max - 5, 3), radius=1, fill=True, color='black'))
    # ax[1].add_patch(Circle((-x_max - 5, 3), radius=1, fill=False, color='green'))
    # ax[1].text(-x_max - 3.5, 2.3, 'target', color='black', fontsize=8)

    plt.savefig(args.out, bbox_inches="tight")
    # save svg for editing
    plt.savefig(
        args.out.replace(".png", ".pdf"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


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
