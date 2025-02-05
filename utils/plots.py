from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from colour import Color
from matplotlib import patches

import wandb
from utils.utils_scripts import get_center

plt.switch_backend("agg")


def distance_bins_plot(
    all_errors: np.ndarray, all_true: np.ndarray, acc: float, n_bins: int = 10
) -> None:
    max_dist = all_true.max()
    if len(all_errors) == 0 or len(all_true) == 0:
        return
    if np.isnan(all_errors).any() or np.isnan(all_true).any():
        return
    bins = np.arange(0, max_dist + n_bins, n_bins)
    hist = []
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_errors = np.asarray(all_errors)[
            (np.asarray(all_true) >= bin_min) & (np.asarray(all_true) < bin_max)
        ]
        if len(bin_errors) == 0:
            hist.append((-1, 0, 1))
            continue
        hist.append((bin_errors.mean(), bin_errors.std(), len(bin_errors)))
    fig, ax = plt.subplots()
    # treat_axis(ax)
    ax.errorbar(
        bins[:-1],
        [h[0] for h in hist],
        yerr=[h[1] for h in hist],
        fmt="o",
        linewidth=2,
        capsize=6,
    )
    for i in range(len(bins) - 1):
        ax.text(
            bins[i] - 3,
            hist[i][0],
            f"${hist[i][0]:.1f}\\pm{hist[i][1]:.1f}$",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Average prediction error (m)")
    ax.set_xlabel("True Distance bins (m)-(m)\n(samples in bin)")
    ax.set_xticks(
        bins[:-1],
        [
            f"{int(bins[i])}-{int(bins[i + 1])}\n({hist[i][2]})"
            for i in range(len(bins) - 1)
        ],
    )
    ax.grid(True)
    ax.set_title(f"Average Prediction Error per Distance Bin (ACC: {acc:.2%})")
    ax.axhline(y=1, color="r", linestyle="-")
    ax.set_ylim(bottom=0)
    ax.plot(bins[:-1], [h[0] for h in hist], color="orange", linewidth=1)
    fig.subplots_adjust(bottom=0.20)

    wandb.log({"Average prediction error": wandb.Image(fig)})
    plt.close(fig)


def bias_plot(pred_dists: list, true_dists: list):
    p = np.asarray(pred_dists)
    t = np.asarray(true_dists)
    if len(p) == 0 or len(t) == 0:
        return
    if np.isnan(p).any() or np.isnan(t).any():
        return
    M = max(p.max(), t.max())
    bias = np.mean(p - t)
    fig, ax = plt.subplots()
    ax.scatter(true_dists, pred_dists, s=1, color="tab:blue")
    ax.set_ylabel("Predicted Distance (m)")
    ax.set_xlabel("True Distance (m)")
    ax.set_title(f"Predicted Distance vs True Distance (bias: {bias:.2f}m)")
    ax.plot([0, M], [0, M], color="tab:red", linestyle="--", linewidth=1)

    ax.set_xlim(left=-0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # set horizontal grid
    ax.yaxis.grid(True, which="major", linestyle="--", color="tab:grey", alpha=0.5)

    wandb.log({"Prediction error vs True Distance": wandb.Image(fig)})
    # plt.savefig('bias_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def visibility_bins_plot(
    all_visibilities: list, all_errors: list, acc: float, n_bins: int = 10
):
    if len(all_errors) == 0 or len(all_visibilities) == 0:
        return
    if np.isnan(all_errors).any() or np.isnan(all_visibilities).any():
        return
    bins = np.arange(0, 1.1, n_bins / 100)
    hist = []
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_errors = np.asarray(all_errors)[
            (np.asarray(all_visibilities) >= bin_min)
            & (np.asarray(all_visibilities) < bin_max)
        ]
        if len(bin_errors) == 0:
            hist.append((-1, 0, 1))
            continue
        hist.append((bin_errors.mean(), bin_errors.std(), len(bin_errors)))
    fig, ax = plt.subplots()
    # treat_axis(ax)
    ax.errorbar(
        bins[:-1],
        [h[0] for h in hist],
        yerr=[h[1] for h in hist],
        fmt="o",
        linewidth=2,
        capsize=6,
    )
    for i in range(len(bins) - 1):
        ax.text(
            bins[i] - 0.03,
            hist[i][0],
            f"${hist[i][0]:.1f}\\pm{hist[i][1]:.1f}$",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Average prediction error (m)")
    ax.set_xlabel("Visibility bins (samples in bin)")
    ax.set_xticks(
        bins[:-1],
        [
            f"{int(bins[i] * 100)}-{int(bins[i + 1] * 100)}\n({hist[i][2]})"
            for i in range(len(bins) - 1)
        ],
    )
    ax.grid(True)
    ax.set_title(f"Average Prediction Error per Visibility Bin (ACC: {acc:.2%})")
    ax.axhline(y=1, color="r", linestyle="-")
    ax.set_ylim(bottom=0)
    ax.plot(bins[:-1], [h[0] for h in hist], color="orange", linewidth=1)
    fig.subplots_adjust(bottom=0.20)

    wandb.log({"Average prediction error per visibility bin": wandb.Image(fig)})
    plt.close(fig)


def average_variance_plot(all_vars: list, all_true: list):
    if len(all_vars) == 0 or len(all_true) == 0:
        return
    if np.isnan(all_vars).any() or np.isnan(all_true).any():
        return
    all_vars = np.asarray(all_vars)
    all_true = np.asarray(all_true)
    all_vars = all_vars[all_true < 100]
    all_true = all_true[all_true < 100]
    bins = np.arange(0, 70 + 10, 10)
    hist = []
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_vars = all_vars[(all_true >= bin_min) & (all_true < bin_max)]
        if len(bin_vars) == 0:
            hist.append((-1, 0, 1))
            continue
        hist.append((bin_vars.mean(), bin_vars.std(), len(bin_vars)))

    fig, ax = plt.subplots()
    ax.errorbar(
        bins[:-1],
        [h[0] for h in hist],
        yerr=[h[1] for h in hist],
        fmt="o",
        linewidth=2,
        capsize=6,
    )
    for i in range(len(bins) - 1):
        ax.text(
            bins[i] - 3,
            hist[i][0],
            f"${hist[i][0]:.1f}\\pm{hist[i][1]:.1f}$",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Average variance (m)")
    ax.set_xlabel("True Distance bins (m)-(m)\n(samples in bin)")
    ax.set_xticks(
        bins[:-1],
        [
            f"{int(bins[i])}-{int(bins[i + 1])}\n({hist[i][2]})"
            for i in range(len(bins) - 1)
        ],
    )
    ax.grid(True)
    ax.set_title("Average Variance per Distance Bin")
    ax.plot(bins[:-1], [h[0] for h in hist], color="orange", linewidth=1)
    fig.subplots_adjust(bottom=0.20)

    wandb.log({"Average variance per distance bin": wandb.Image(fig)})
    plt.close(fig)


def torch_img_to_numpy(img: torch.Tensor) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()
        if img.shape[0] == 4:
            img = img[:3]
        if img.ndim == 4:
            img = img[:, -1]
        img = np.transpose(img, (1, 2, 0))
    return img


def qualitative_with_centers(
    x: np.ndarray,
    bboxes,
    distances_true,
    distances_pred,
    distances_vars,
    visibilities,
    step: int,
) -> None:
    centers = [get_center(bb) for bb in bboxes[0]]
    centers = [(int(round(c[0])), int(round(c[1]))) for c in centers]

    if isinstance(distances_true, torch.Tensor):
        distances_true = distances_true.detach().cpu().numpy().tolist()
    if isinstance(distances_pred, torch.Tensor):
        distances_pred = distances_pred.detach().cpu().numpy().tolist()
    if isinstance(distances_vars, torch.Tensor):
        distances_vars = distances_vars.detach().cpu().cpu().numpy().tolist()
    if isinstance(visibilities, torch.Tensor):
        visibilities = visibilities.detach().cpu().numpy().tolist()

    fig, ax = plt.subplots()
    x = torch_img_to_numpy(x)
    ax.imshow(x)

    for j in range(len(centers)):
        center = centers[j]
        # draw circle of center
        ax.add_patch(plt.Circle(center, 2, color="tab:red", fill=True))

        # draw text of true distance at center
        dist = distances_true[j]

        ax.text(
            center[0] + 20,
            center[1],
            f"{dist:.1f}",
            color="tab:green",
            ha="center",
            va="center",
        )

        # draw text of predicted distance at center
        dist = distances_pred[j]
        ax.text(
            center[0] + 20,
            center[1] + 12,
            f"{dist:.1f}",
            color="tab:red",
            ha="center",
            va="center",
        )

        # draw text of visibility at center
        dist = 1 - visibilities[j]
        ax.text(
            center[0] + 20,
            center[1] + 24,
            f"occ {dist:.0%}",
            color="tab:orange",
            ha="center",
            va="center",
        )

        if distances_vars:
            # draw text of variance at center
            dist = distances_vars[j]
            ax.text(
                center[0] + 20,
                center[1] + 36,
                f"var {dist:.1f}",
                color="tab:blue",
                ha="center",
                va="center",
            )

    wandb.log({f"Qualitative with centers {step}": wandb.Image(fig)})
    plt.close(fig)


def draw_outliers_bb(step, x, last_frame_distances, last_frame_bboxes, dists_pred):
    outliers = [d for d in dists_pred if d < 1]
    if len(outliers) > 0:
        o = np.asarray(dists_pred)
        idxs = np.where(o < 1)[0]
        x_img = x[0, :3, -1].cpu().detach().numpy()
        x_img = np.moveaxis(x_img, 0, -1)
        fig, ax = plt.subplots(1)
        for i in idxs:
            bbox = last_frame_bboxes[0][i].tolist()
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                bbox[0], bbox[1], f"{dists_pred[i]:.2f}m", color="tab:red", size="small"
            )
            ax.text(
                bbox[0],
                bbox[1] + 10,
                f"{last_frame_distances[i]:.2f}m",
                color="tab:green",
                size="small",
            )
            # draw coords on corners of bbox
            ax.text(
                bbox[0] - 5,
                bbox[1] + 5,
                f"{bbox[0]:.0f},{bbox[1]:.0f}",
                color="tab:orange",
                size="small",
            )
            ax.text(
                bbox[2] + 5,
                bbox[1] + 5,
                f"{bbox[2]:.0f},{bbox[1]:.0f}",
                color="tab:orange",
                size="small",
            )
            ax.text(
                bbox[0] - 5,
                bbox[3] - 5,
                f"{bbox[0]:.0f},{bbox[3]:.0f}",
                color="tab:orange",
                size="small",
            )
            ax.text(
                bbox[2] + 5,
                bbox[3] - 5,
                f"{bbox[2]:.0f},{bbox[3]:.0f}",
                color="tab:orange",
                size="small",
            )

        ax.imshow(x_img)
        ax.axis("off")
        plt.savefig(f"outlier_{step}.png", bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close(fig)


def draw_bb(x, bboxes, dists, fname):
    x_img = x[:3].cpu().detach().numpy()
    x_img = np.moveaxis(x_img, 0, -1)
    fig, ax = plt.subplots(1)
    H, W = x_img.shape[:2]
    for bbox, d in zip(bboxes, dists):
        x1, y1, x2, y2 = bbox.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        # x_img = cv2.rectangle(x_img.copy(), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        # ax.text(x1, y1, f'{d:.2f}m', color='tab:green', size='small')
        # draw cord of top left corner
        ax.text(x1 - 5, y1 + 5, f"y:{y1:.0f}", color="tab:orange")
        if not (0 <= x1 <= W and 0 <= x2 <= W and 0 <= y1 <= H and 0 <= y2 <= H):
            print("out of bounds ->", x1, x2, y1, y2, W, H)
        # break

    ax.imshow(x_img)
    # ax.axis('off')
    plt.savefig(f"imgs/{fname}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def draw_kitti_bb(x, bboxes, dists, fname, gt_dist=None):
    x_img = x[:3].cpu().detach().numpy()
    x_img = np.moveaxis(x_img, 0, -1)
    fig, ax = plt.subplots(1)
    H, W = x_img.shape[:2]
    for i, (bbox, d) in enumerate(zip(bboxes[0].tolist(), dists)):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        # x_img = cv2.rectangle(x_img.copy(), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        # ax.text(x1, y1, f'{d:.2f}m', color='tab:green', size='small')
        # draw cord of top left corner
        ax.text(x1 - 5, y1 + 5, f"d:{d:.0f}", color="tab:orange")
        ax.text(
            x1 - 5, y1 + 10, f"d_gt:{gt_dist[i]:.0f}", color="tab:red"
        ) if gt_dist is not None else None
        if not (0 <= x1 <= W and 0 <= x2 <= W and 0 <= y1 <= H and 0 <= y2 <= H):
            print("out of bounds ->", x1, x2, y1, y2, W, H)
        # break

    ax.imshow(x_img)
    # ax.axis('off')
    plt.savefig(f"imgs/{fname}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def draw_bb_torch(x, bboxes, dists, fname):
    x_img = x[:3].cpu().detach()  # .numpy()
    x_img = (x_img * 255).type(torch.uint8)
    if len(bboxes) > 0:
        x_img = torchvision.utils.draw_bounding_boxes(
            x_img, bboxes, labels=[str(round(x.item(), 2)) for x in dists]
        )
    torchvision.utils.save_image(x_img / 255.0, f"imgs/{fname}.png")


def draw_image_with_true_and_pred(img, bbobxes, y_true, y_pred, fname):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")
    errors = [abs(y_pred[i] - y_true[i]) for i in range(len(y_true))]
    colors = [
        color.hex
        for color in Color("green").range_to(Color("red"), int(max(errors)) + 2)
    ]
    for i, bbox in enumerate(bbobxes):
        x1, y1, x2, y2 = bbox
        ax.text(x1, y1, f"{y_pred[i]:.2f}m", color="tab:red", size="small", stroke=1)
        ax.text(x1, y1 + 30, f"GT:{y_true[i]:.2f}m", color="tab:green", size="small")
        # bb color green to red gradient based on error
        bb_color = colors[int(round(errors[i]))]
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor=bb_color,
            facecolor="none",
        )
        ax.add_patch(rect)

    Path("imgs").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"imgs/{fname}.png", bbox_inches="tight", pad_inches=0.0, dpi=600)
    plt.close(fig)
