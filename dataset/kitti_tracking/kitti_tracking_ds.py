import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from matplotlib import patches
from torch.utils.data import Dataset

from utils import gt
from utils.transforms import (
    ComposeBB,
    ConvertBCHWtoCBHW,
    ConvertBHWCtoBCHW,
    ConvertImageDtypeBB,
    NoisyBB,
    NoisyBBIouPreserving,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensorBB,
)
from utils.utils_scripts import z_to_nearness

MAX_FRAME_SIZE = (375, 1242)
COLUMNS = ("fid tid type trunc occ alpha x1 y1 x2 y2 " "h w l x y z ry score").split()


class KittiTracking(Dataset):
    def __init__(
        self, cnf: argparse.Namespace, root_dir: str = "data/kitti_tracking", **kwargs
    ):
        self.root_dir = root_dir
        self.mode = "test"
        self.cnf = cnf
        self.class_to_int = {
            "Car": 0,
            "Van": 1,
            "Truck": 2,
            "Pedestrian": 3,
            "Person_sitting": 4,
            "Cyclist": 5,
            "Tram": 6,
            "Misc": 7,
            "DontCare": 8,
        }
        self.transform_bb, self.transform_clip = self.get_transforms()

        all_seq = sorted(os.listdir(os.path.join(root_dir, "training/image_02")))
        self.sequences = {}
        self.labels = {}
        for seq in tqdm(all_seq, desc="Loading Kitti Tracking dataset"):
            seq_dir = os.path.join(root_dir, "training/image_02", seq)
            self.sequences[seq] = sorted(Path(seq_dir).glob("*.png"))
            seq_gt_path = os.path.join(root_dir, "training/label_02", seq + ".txt")
            seq_gt = pd.read_csv(
                seq_gt_path, header=None, sep=" ", names=COLUMNS, skip_blank_lines=True
            )
            seq_gt = seq_gt[seq_gt["type"] != "DontCare"]
            for frame in seq_gt["fid"].unique():
                # select class_names(0), occlusion(2), bbox (4, 5, 6, 7), locations (11, 12, 13)
                frame_gt = seq_gt[seq_gt["fid"] == frame][
                    "type occ x1 y1 x2 y2 x y z".split()
                ]
                frame_gt["type"] = frame_gt["type"].map(self.class_to_int)

                self.labels[(seq, frame)] = frame_gt.values

        self.index_to_label = {i: label for i, label in enumerate(sorted(self.labels))}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq, frame = self.index_to_label[idx]
        labels = self.labels[(seq, frame)]

        classes = [labels[:, 0]]
        occlusions = torch.from_numpy(labels[:, 1])
        visibilities = [
            torch.ones(len(occlusions), dtype=torch.float32) - (occlusions / 3)
        ]
        video_bboxes = [torch.from_numpy(labels[:, 2:6])]
        coords_3D = [torch.from_numpy(labels[:, 9:12])]
        video_dists = [labels[:, -1]]

        clip = np.asarray(Image.open(self.sequences[seq][frame]))[None, ...]

        good_idxs = [
            torch.ones(len(video_bboxes[i]), dtype=torch.bool)
            for i in range(len(video_bboxes))
        ]

        clip, video_bboxes, video_dists, good_idxs = self.transform_bb(
            clip, video_bboxes, video_dists, good_idxs
        )

        c_maps = torch.tensor([])

        for i, img in enumerate(clip):
            if len(video_bboxes[i]) == 0:
                c_maps = torch.cat((c_maps, torch.zeros((1, *self.input_size))), dim=0)
                continue

            visibilities[i] = visibilities[i][good_idxs[i]]
            coords_3D[i] = coords_3D[i][good_idxs[i]]

            d = z_to_nearness(video_dists[i]) if self.cnf.nearness else video_dists[i]
            c_map, *_ = gt.get_gt(
                img=img,
                bboxes=video_bboxes[i],
                dists=d,
                sigma=self.cnf.sigma,
                scale_factor=self.cnf.scale_factor,
                ds_stats=self.cnf.ds_stats,
                radius=self.cnf.radius,
            )

            if self.cnf.use_centers:
                c_maps = torch.cat(
                    (c_maps, torch.tensor(c_map, dtype=torch.float).unsqueeze(0)), dim=0
                )

        clip = self.transform_clip(clip)
        clip_clean = clip
        tracking_ids = []

        if self.cnf.use_centers:
            clip = torch.cat((clip, c_maps.unsqueeze(0)), dim=0)

        return (
            clip,
            [],
            [],
            video_bboxes,
            video_dists,
            visibilities,
            classes,
            coords_3D,
            [frame],
            0,
            tracking_ids,
            clip_clean,
        )

    def get_transforms(self):
        if self.mode == "train":
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    RandomHorizontalFlip(p=0.5),
                ]
            )

            if self.cnf.noisy_bb:
                T_BB.transforms.append(NoisyBB())

            if self.cnf.noisy_bb_with_iou:
                T_BB.transforms.append(NoisyBBIouPreserving(self.cnf.noisy_bb_iou_th))

            if self.cnf.random_crop:
                T_BB.transforms.append(
                    RandomCrop(self.cnf.input_h_w, self.cnf.crop_mode)
                )

            T_BB.transforms.append(Resize(MAX_FRAME_SIZE))

        else:
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    Resize(MAX_FRAME_SIZE),
                ]
            )

        T_CLIP = ConvertBCHWtoCBHW()
        return T_BB, T_CLIP

    @staticmethod
    def collate_fn(batch):
        clip, *other, clip_clean = zip(*batch)
        return torch.stack(clip), *other, torch.stack(clip_clean)

    @staticmethod
    def wif(worker_id: int) -> None:
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = worker_id + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def wif_test(worker_id):
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = worker_id + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


if __name__ == "__main__":

    def draw(x, bboxes, dists, fname, gt_dist=None):
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
        plt.savefig(f"{fname}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    import main

    args = main.parse_args()
    ds = KittiTracking(args)
    x = ds[0]
    draw(x[0].squeeze(), x[3], x[4][0], "a")
    # print(x)
