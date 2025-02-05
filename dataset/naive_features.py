import argparse
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import torch
from torch import Tensor

from dataset.video_dataset import VideoFrameDataset
from utils.utils_scripts import Timer, get_center


class NaiveFeatures(VideoFrameDataset):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf)
        self.FILENAME_LEN = 4
        self.ANNOTATIONS_FILE = Path(
            f"dataset/motsynth/motsynth_splits/motsynth_{mode}.txt"
        )
        self.MODE = mode
        self.MAX_STRIDE = 1
        self.clip_len = 1
        self.min_visibility = self.cnf.min_visibility
        self.class_to_int = {"Pedestrian": 0}

        with Timer(f"Loading {mode} annotations", one_line=True):
            self.sequences = tuple(self.ANNOTATIONS_FILE.read_text().splitlines())
            self.annotations = {
                seq: np.load(Path(self.annotations_path, f"{seq}.npy"))
                for seq in self.sequences
            }

        with Timer(f"Loading {mode} videos", one_line=True):
            self._load_videos()

    def get_frames_path(self, video: str) -> str:
        return f"frames/{video}/rgb"

    def get_labels(self, idx: int) -> Tuple[np.ndarray, int]:
        video_idx = np.searchsorted(self.cumulative_idxs, idx, side="right") - 1
        starting_frame = idx - self.cumulative_idxs[video_idx]
        video_name, frames_name = self.videos[video_idx].get_clip(
            starting_frame, only_labels=True
        )
        labels = self.annotations[video_name]
        labels = [
            labels[
                (labels[:, 0] == float(frame))
                & (labels[:, 5] == 1.0)
                & (labels[:, 8] >= self.min_visibility)
            ][:, 1:]
            for frame in frames_name
        ]
        return labels[0], video_idx


class MLPDataset(NaiveFeatures):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor, Tensor]:
        labels, video_idx = self.get_labels(idx)

        h_norm = []
        w_norm = []
        ratio = []
        x_rel = []
        y_rel = []
        distances = []
        visibilities = []

        for i, bbox in enumerate(labels):
            if len(bbox) == 0:
                continue
            x1, y1, x2, y2, _, _, dist, visibility, *_ = bbox
            h = y2 - y1
            w = x2 - x1
            H, W = self.videos[video_idx].frame_size
            h_norm.append(h / H)
            w_norm.append(w / W)
            ratio.append(h / w)
            c_x, c_y = get_center(bbox[:5])
            x_rel.append((c_x / W))
            y_rel.append((c_y / H))
            distances.append(dist)
            visibilities.append(visibility)

        x = torch.tensor([h_norm, w_norm, ratio, x_rel, y_rel], dtype=torch.float32).T
        distances = torch.tensor(distances, dtype=torch.float32)
        visibilities = torch.tensor(visibilities, dtype=torch.float32)
        classes = torch.zeros_like(distances)

        return x, distances, visibilities, classes


class DisNetDataset(NaiveFeatures):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf, mode)
        self.person_prior = torch.tensor([1.75, 0.55, 0.30]).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor, Tensor]:
        labels, video_idx = self.get_labels(idx)

        h_norm = []
        w_norm = []
        d_norm = []
        distances = []
        visibilities = []
        H, W = self.videos[video_idx]._get_frame_size()
        DIAG = np.sqrt(H**2 + W**2)

        for i, bbox in enumerate(labels):
            if len(bbox) == 0:
                continue
            x1, y1, x2, y2, _, _, dist, visibility, *_ = bbox
            h = y2 - y1
            w = x2 - x1

            h_norm.append(H / h)
            w_norm.append(W / w)
            d_norm.append(DIAG / np.sqrt(h**2 + w**2))

            distances.append(dist)
            visibilities.append(visibility)

        x = torch.tensor([h_norm, w_norm, d_norm]).T
        # concat person prior
        x = torch.cat((x, self.person_prior.repeat(x.shape[0], 1)), dim=1).type(
            torch.float32
        )

        distances = torch.tensor(distances)
        visibilities = torch.tensor(visibilities)
        classes = torch.zeros_like(distances)

        return x, distances, visibilities, classes


class SVRDataset(NaiveFeatures):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor, Tensor]:
        labels, video_idx = self.get_labels(idx)

        hs = []
        ws = []
        distances = []
        visibilities = []
        for i, bbox in enumerate(labels):
            if len(bbox) == 0:
                continue
            x1, y1, x2, y2, _, _, dist, visibility, *_ = bbox
            h = y2 - y1
            w = x2 - x1
            hs.append(h)
            ws.append(w)
            distances.append(dist)
            visibilities.append(visibility)

        x = torch.tensor([hs, ws]).T
        distances = torch.tensor(distances)
        visibilities = torch.tensor(visibilities)
        classes = torch.zeros_like(distances)

        return x, distances, visibilities, classes
