import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.video_dataset import VideoFrameDataset
from scripts.mot17_info import MOT17_TEST_SEQUENCES, MOT17_TRAIN_SEQUENCES
from scripts.mot20_info import MOT20_TEST_SEQUENCES, MOT20_TRAIN_SEQUENCES
from utils import gt
from utils.transforms import ConvertBCHWtoCBHW
from utils.utils_scripts import Timer


class MOTChallengeInfer(VideoFrameDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        detections_th=0.1,
        return_only_clean=False,
        no_transforms=False,
    ) -> None:
        super().__init__(args)
        self.version = args.infer_mot_version
        assert self.version in {"17", "20"}, "version must be '17' or '20'"

        self.return_only_clean = return_only_clean

        self.FILENAME_LEN = 6
        self.video_path = Path(self.cnf.mot_challenge_path, f"MOT{self.version}")
        self.SPLIT = 1
        self.MODE = "test"
        self.MAX_STRIDE = 1

        self.no_transforms = no_transforms

        self.split = args.infer_split
        assert self.split in {"train", "test"}, "split must be 'train' or 'test'"
        if self.version == "17":
            selected_sequences = (
                MOT17_TRAIN_SEQUENCES if self.split == "train" else MOT17_TEST_SEQUENCES
            )
        else:
            selected_sequences = (
                MOT20_TRAIN_SEQUENCES if self.split == "train" else MOT20_TEST_SEQUENCES
            )

        self.infer_gt = args.infer_gt if hasattr(args, "infer_gt") else False
        self.name = f"MOT{self.version}_{self.split}_{'gt' if self.infer_gt else 'det'}"

        self.detections_th = detections_th

        assert (
            self.cnf.detections_path is not None
        ), "MOTChallenge dataset requires detections_path to be set"

        det_path = Path(self.cnf.detections_path)
        with Timer(f"Loading detections from {det_path}"):
            # get extension
            if det_path.suffix == ".npz":
                self.annotations = {
                    k: v
                    for k, v in np.load(det_path).items()
                    if k in selected_sequences
                }
            elif det_path.is_dir():
                npys = list(det_path.glob("*.npy"))
                self.annotations = {
                    npy.stem: np.load(npy)
                    for npy in npys
                    if npy.stem in selected_sequences
                }

        assert len(self.annotations) > 0, f"No detections found in {det_path}"
        assert len(self.annotations) == len(selected_sequences), (
            f"Number of detections ({len(self.annotations)}) does not match "
            f"number of sequences ({len(selected_sequences)})"
        )

        self.sequences = self.annotations.keys()

        with Timer(f"Loading videos from {self.video_path}"):
            self._load_videos()

        # get all if ground truth (training), otherwise get second part (validation)
        # for video in self.videos:
        #     if not args.infer_gt and self.split == 'train' and self.version == '17':
        #         video.frame_list = video.frame_list[len(video.frame_list) // 2 + 1:]

        self.cumulative_idxs = np.cumsum([0] + [len(v) for v in self.videos])

    def get_frames_path(self, video: str) -> str:
        return f"{self.split}/{video}/img1"

    def __getitem__(self, idx: int):
        video_idx = np.searchsorted(self.cumulative_idxs, idx, side="right") - 1
        starting_frame = idx - self.cumulative_idxs[video_idx]
        clip, video_name, frames_name = self.videos[video_idx].get_clip(starting_frame)

        if self.return_only_clean:
            clip = torch.tensor(clip) if isinstance(clip, np.ndarray) else clip
            clip = clip.permute(0, 3, 1, 2)
            if self.no_transforms:
                return clip, frames_name, video_name
            if self.MODE == "train":
                crop = transforms.RandomResizedCrop(
                    self.input_size,
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                )
            else:
                crop = transforms.Compose(
                    [
                        transforms.Resize(
                            self.input_size[0], interpolation=InterpolationMode.BICUBIC
                        ),
                        transforms.CenterCrop(self.input_size),
                    ]
                )

            T = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float32),
                    crop,
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ConvertBCHWtoCBHW(),
                ]
            )
            return T(clip), frames_name, video_name

        labels = self.annotations[video_name]
        # select labels where the first element is equal to the frame name
        detections_labels = [
            labels[labels[:, 0] == float(frame)][:, 1:] for frame in frames_name
        ]

        # x1, y1, x2, y2, (4) valid, (5) tracking_id, (6) dist, (7) visibility, x, y, z

        scores = [bboxes[:, 4] for bboxes in detections_labels]
        detections = [bboxes[:, 0:4] for bboxes in detections_labels]

        good_detections_idx = [np.where(s > self.detections_th)[0] for s in scores]
        detections = [
            detections[i][good_detections_idx[i]] for i in range(len(detections))
        ]
        scores = [scores[i][good_detections_idx[i]] for i in range(len(scores))]
        if self.infer_gt:
            if detections_labels[0].shape[1] < 8:
                raise ValueError(
                    "No tracking_id(5), dist(6), visibility(7) in detections"
                )
            trackids = [bboxes[:, 5] for bboxes in detections_labels]
            trackids = [
                trackids[i][good_detections_idx[i]] for i in range(len(trackids))
            ]
            gt_dists = [bboxes[:, 6] for bboxes in detections_labels]
            gt_dists = [
                gt_dists[i][good_detections_idx[i]] for i in range(len(gt_dists))
            ]
            vis = [bboxes[:, 7] for bboxes in detections_labels]
            vis = [vis[i][good_detections_idx[i]] for i in range(len(vis))]

        detections_raw = (
            detections.copy()
        )  # detections_raw is a copy. detections will be cropped-resized etc.

        clip, detections, _ = self.augmenter.apply_video(
            clip=clip, bboxes=detections, dists=[], mode=self.MODE
        )

        detections = [torch.tensor(bboxes) for bboxes in detections]

        c_maps = torch.tensor([])

        for i, img in enumerate(clip):
            if len(detections[i]) == 0 and self.cnf.use_centers:
                c_maps = torch.cat((c_maps, torch.zeros((1, *self.input_size))), dim=0)
                continue

            if self.cnf.use_centers:
                c_map = gt.get_gt_cmap_only(
                    img=img, bboxes=detections[i], sigma=self.cnf.sigma
                )

                c_maps = torch.cat(
                    (c_maps, torch.tensor(c_map, dtype=torch.float).unsqueeze(0)), dim=0
                )

        clip = self.transforms(torch.tensor(np.stack(clip).transpose((3, 0, 1, 2))))

        if self.cnf.use_centers:
            clip = torch.cat((clip, c_maps.unsqueeze(0)), dim=0)

        if self.infer_gt:
            return (
                clip,
                detections,
                detections_raw,
                scores,
                trackids,
                gt_dists,
                vis,
                video_name,
                frames_name[-1],
            )
        return clip, detections, detections_raw, scores, video_name, frames_name[-1]

    def collate_fn(self, batch):
        clip, *other = zip(*batch)
        return torch.stack(clip), *other

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
