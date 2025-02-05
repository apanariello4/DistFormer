import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import PIL
import torch
from scipy.stats import truncnorm
from torch.utils.data import Dataset
from torchvision import transforms

from utils.bboxes import draw_bbox
from utils.transforms import ConvertBCHWtoCBHW, ConvertBHWCtoBCHW
from utils.utils_scripts import colors, get_center, imread

try:
    import ujson as json
except ImportError:
    import json


class VideoFrames:
    def __init__(
        self,
        path: Path,
        clip_len: int,
        mode: str,
        max_stride: int,
        stride_sampling: str,
        filename_len: int,
        sampling: str,
        end_to: Optional[int],
    ):
        self.path = path
        self.clip_len = clip_len
        self.mode = mode
        self.max_stride = max_stride
        self.filename_len = filename_len
        self.stride_sampling = stride_sampling
        self.video_name = self.path.parent.name
        self.sampling = sampling

        # check if 'cache/{path}.json' exists
        legal_path = str(self.path).replace("/", "_").replace("\\", "_")
        cache_path = Path("cache", f"{legal_path}.json")
        if cache_path.exists():
            with cache_path.open("r") as f:
                self.frame_list = tuple(json.load(f))
        else:
            # create cache
            self.frame_list = tuple(sorted(os.listdir(self.path)))
            cache_path.parent.mkdir(exist_ok=True)
            with cache_path.open("w") as f:
                json.dump(self.frame_list, f)
        # self.frame_list = tuple(sorted(os.listdir(self.path)))

        self.frame_list = self.frame_list[:end_to]
        # if 'MOT17' in self.video_name:
        #     self.frame_list = self.frame_list[len(self.frame_list) // 2 + 1:]

        # self.frame_size = self._get_frame_size()

    def _get_frame_size(self) -> Tuple[int, int]:
        return imread(str(Path(self.path, self.frame_list[0]))).shape[:2]

    @staticmethod
    def fast_loading_frames(frames):
        return np.array([np.asarray(PIL.Image.open(f)) for f in frames])

    def get_clip(self, idx: int, only_labels: bool = False):
        # assert 0 <= idx * self.split < len(self), f'index {idx * self.split} out of range'
        stride = self.get_stride()

        # Get indices
        if self.sampling == "smart":
            idx_range = self.smart_frame_replication(idx, self.clip_len, stride)
        else:
            idx_range = [max(0, idx - i * stride) for i in range(self.clip_len)][::-1]

        frames_name = [self.remove_extension(self.frame_list[i]) for i in idx_range]

        if not only_labels:
            # Load clip
            selected_frame_paths = [
                str(Path(self.path, self.frame_list[i])) for i in idx_range
            ]
            clip = self.fast_loading_frames(selected_frame_paths)
            return clip, self.video_name, frames_name

        return self.video_name, frames_name

    def __len__(self) -> int:
        return len(self.frame_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(seq="{self.video_name}", len={self.clip_len}, mode={self.mode}, stride={self.max_stride})'

    def remove_extension(self, filename: str):
        return str(Path(filename).stem)

    def get_stride(self) -> int:
        assert self.max_stride > 0, "max_stride must be > 0"
        distribution = self.stride_sampling

        if self.max_stride == 1:
            return 1
        if self.mode == "test" and distribution != "fixed":
            return self.max_stride // 2

        if distribution == "normal":
            lower, upper = 1, self.max_stride
            mu = self.max_stride / 2
            sigma = self.max_stride / 5
            stride = np.round(
                truncnorm(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
                ).rvs()
            ).astype(int)
            # stride = np.random.normal(self.max_stride // 2, self.max_stride // 2)
            # stride = int(np.rint(np.clip(stride, 1, self.max_stride))) if self.mode == "train" else self.max_stride // 2
        elif distribution == "uniform":
            stride = random.randint(1, self.max_stride)
        elif distribution == "fixed":
            stride = self.max_stride
        else:
            raise ValueError(f"Unknown distribution {distribution}")

        return 1 if stride == 0 else stride

    @staticmethod
    def smart_frame_replication(idx: int, clip_len: int, stride: int) -> List[int]:
        """When not enough frames are available from idx backward, replicate the frames smartly.
        If two frames are available i.e. idx=1, replicate half the time the first and half the time the last frame
        and so on. When more than clip_len frames are available, but less than stride*clip_len, apply a linspace,
        to get the highest possible coverage from 0 to idx.

        return idxs of frames to be used
        """
        if idx < clip_len * stride:
            # not enough frames available
            if idx == 0:
                idxs = [0] * clip_len
            elif idx < clip_len:
                repl_each_frame = clip_len // (idx + 1)
                repl_last_frame = clip_len % (idx + 1)
                idxs = []
                for i in range(idx):
                    idxs += [i] * repl_each_frame
                idxs += [idx] * (repl_each_frame + repl_last_frame)

            elif idx >= clip_len:
                # sample with equidistance with the highest possible coverage from 0 to idx
                idxs = np.linspace(0, idx, clip_len, dtype=int).tolist()
        else:
            idxs = [idx - i * stride for i in range(clip_len)][::-1]
        return idxs


class VideoFrameDataset(Dataset):
    FILENAME_LEN: int
    MODE: str
    MAX_STRIDE: int
    ANNOTATIONS_FILE: Path

    def __init__(self, cnf: argparse.Namespace) -> None:
        self.cnf = cnf
        self.clip_len = self.cnf.clip_len
        assert self.clip_len > 0, "clip_len must be > 0"
        self.video_path = self.cnf.ds_path
        self.annotations_path = self.cnf.annotations_path
        self.stride_sampling = self.cnf.stride_sampling
        self.input_size = self.cnf.input_h_w
        self.sampling = self.cnf.sampling
        self.END_TO = None

        self.transforms = transforms.Compose(
            [
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                ConvertBCHWtoCBHW(),
            ]
        )

    def _load_annotations(self) -> None:
        raise DeprecationWarning("Deprecated")

    def _load_videos(self) -> None:
        v = self.sequences if hasattr(self, "sequences") else self.annotations

        self.videos = tuple(
            VideoFrames(
                Path(self.video_path, self.get_frames_path(video)),
                self.clip_len,
                self.MODE,
                self.MAX_STRIDE,
                self.stride_sampling,
                self.FILENAME_LEN,
                self.sampling,
                self.END_TO,
            )
            for video in v
        )

        self.cumulative_idxs = np.cumsum([0] + [len(v) for v in self.videos])

    def __len__(self) -> int:
        return self.cumulative_idxs[-1]

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def get_frames_path(self, video: str) -> str:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        return None

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
    import cv2

    import main
    from dataset.motsynth import MOTSynth

    cnf = main.parse_args()
    cnf.use_debug_dataset = False
    cnf.annotations_path = "./dataset"
    dataset = MOTSynth(cnf, mode="train")
    print(len(dataset))
    print(dataset[10])

    clip, h_maps, w_maps, d_map, m_map, video_bboxes, video_dists, visibilities = (
        dataset[0]
    )

    clip = (clip[:-1].permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8).copy()

    for x, frame in enumerate(clip):
        for i, bbox in enumerate(video_bboxes[x]):
            bb = bbox.tolist()
            clip[x] = draw_bbox(
                clip[x], bb, thickness=1, alpha=0, color=colors.FLAT_RED.value
            )
            clip[x] = cv2.putText(
                clip[x],
                f"{video_dists[x][i]:.2f}",
                [int(x) for x in get_center(bb)],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colors.FLAT_RED.value,
                2,
                cv2.LINE_AA,
                False,
            )
    from torchvision.utils import save_image

    save_image(torch.from_numpy(clip).permute(0, 3, 1, 2) / 255, "a.png")
