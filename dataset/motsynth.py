import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from dataset.video_dataset import VideoFrameDataset
from scripts.motsynth_info import (
    MOTSYNTH_TEST_STATIC_SEQUENCES,
    MOTSYNTH_TRAIN_STATIC_CLEAN_120,
    MOTSYNTH_VAL_STATIC_SEQUENCES,
)
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
from utils.utils_scripts import Timer, z_to_nearness


class MOTSynth(VideoFrameDataset):
    def __init__(
        self,
        cnf: argparse.Namespace,
        mode: str = "train",
        return_only_clean: bool = False,
        seq=None,
        no_transforms: bool = False,
    ) -> None:
        super().__init__(cnf)
        self.class_to_int = {"Pedestrian": 0}
        self.FILENAME_LEN = 4
        self.MODE = mode
        self.min_visibility = self.cnf.min_visibility
        self.return_only_clean = return_only_clean
        self.no_transforms = no_transforms
        if self.cnf.use_debug_dataset:
            self.ANNOTATIONS_FILE = Path("dataset/motsynth/motsynth_splits/motsynth_debug.txt")
            self.MAX_STRIDE = 1
        else:
            self.ANNOTATIONS_FILE = Path(f"dataset/motsynth/motsynth_splits/motsynth_{mode}.txt")
            assert (self.ANNOTATIONS_FILE.exists()), f"File {self.ANNOTATIONS_FILE} does not exist"
            self.MAX_STRIDE = self.cnf.max_stride

        self.transform_bb, self.transform_clip = self.get_transforms(self.cnf, self.MODE)

        with Timer(f"Loading {mode} annotations", one_line=True):
            self.sequences = tuple(self.ANNOTATIONS_FILE.read_text().splitlines())
            if seq is not None:
                self.sequences = [seq]
            self.annotations = {seq: np.load(Path(self.annotations_path, f"{seq}.npy")) for seq in self.sequences}

        with Timer(f"Loading {mode} videos", one_line=True):
            self._load_videos()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, list]:
        video_idx, frame_idx = self._get_vid_and_frame_idxs(idx)
        clip, video_name, frames_name = self.videos[video_idx].get_clip(frame_idx)

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
        labels = self._get_labels(video_name, frames_name)

        video_bboxes, tracking_ids, video_dists, visibilities, head_coords = (
            self.extract_gt(labels)
        )

        good_idxs = [
            torch.ones(len(video_bboxes[i]), dtype=torch.bool)
            for i in range(len(video_bboxes))
        ]
        clip, video_bboxes, video_dists, good_idxs = self.transform_bb(
            clip, video_bboxes, video_dists, good_idxs
        )

        c_maps = torch.tensor([])
        d_map = torch.zeros(
            (
                1,
                int(self.input_size[0] * self.cnf.scale_factor),
                int(self.input_size[1] * self.cnf.scale_factor),
            )
        )
        m_map = torch.zeros(
            (
                1,
                int(self.input_size[0] * self.cnf.scale_factor),
                int(self.input_size[1] * self.cnf.scale_factor),
            )
        )

        for i, img in enumerate(clip):
            if len(video_bboxes[i]) == 0:
                c_maps = torch.cat((c_maps, torch.zeros((1, *self.input_size))), dim=0)
                continue

            tracking_ids[i] = tracking_ids[i][good_idxs[i]]
            visibilities[i] = visibilities[i][good_idxs[i]]
            head_coords[i] = head_coords[i][good_idxs[i]]

            # generate target maps: center heatmap, height map, width map
            d = z_to_nearness(video_dists[i]) if self.cnf.nearness else video_dists[i]
            c_map, _, _, d_map_np, m_map_np = gt.get_gt(
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

            if i == len(clip) - 1:
                d_map = torch.tensor(d_map_np, dtype=torch.float).unsqueeze(0)
                m_map = torch.tensor(m_map_np, dtype=torch.float).unsqueeze(0)

        clip_clean = ConvertBCHWtoCBHW()(clip)

        clip = self.transform_clip(clip)

        if self.cnf.use_centers:
            clip = torch.cat((clip, c_maps.unsqueeze(0)), dim=0)
        classes = [torch.full_like(video_dist, 0.0) for video_dist in video_dists]

        return (
            clip,
            d_map,
            m_map,
            video_bboxes,
            video_dists,
            visibilities,
            classes,
            head_coords,
            frames_name,
            video_name,
            tracking_ids,
            None,
            clip_clean,
        )

    def extract_gt(self, labels):
        # x1, y1, x2, y2, valid_flag, track_id, distance, visibility, x, y, z
        video_bboxes = [torch.from_numpy(bboxes[:, :4]) for bboxes in labels]
        tracking_ids = [torch.from_numpy(bboxes[:, 5]) for bboxes in labels]
        video_dists = [bboxes[:, 6] for bboxes in labels]
        visibilities = [torch.from_numpy(bboxes[:, 7]) for bboxes in labels]
        head_coords = [torch.from_numpy(bboxes[:, 8:11]) for bboxes in labels]
        return video_bboxes, tracking_ids, video_dists, visibilities, head_coords

    def _get_labels(self, video_name, frames_name):
        labels = self.annotations[video_name]
        return [
            labels[
                (labels[:, 0] == float(frame))
                & (labels[:, 5] == 1.0)
                & (labels[:, 8] >= self.min_visibility)
            ][:, 1:]
            for frame in frames_name
        ]

    def _get_vid_and_frame_idxs(self, idx):
        video_idx = np.searchsorted(self.cumulative_idxs, idx, side="right") - 1
        starting_frame = idx - self.cumulative_idxs[video_idx]
        return video_idx, starting_frame

    def get_frames_path(self, video: str) -> str:
        return f"frames/{video}/rgb"

    @staticmethod
    def get_transforms(cnf, mode):
        if mode == "train":
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    RandomHorizontalFlip(p=0.5),
                ]
            )

            if cnf.noisy_bb:
                T_BB.transforms.append(NoisyBB())

            if cnf.random_crop:
                min_crop, max_crop = cnf.crop_range
                T_BB.transforms.append(
                    RandomCrop(
                        cnf.input_h_w,
                        crop_mode=cnf.crop_mode,
                        min_crop=min_crop,
                        max_crop=max_crop,
                    )
                )

            T_BB.transforms.append(Resize(cnf.input_h_w, antialias=None))

            T_CLIP = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                            )
                        ],
                        p=0.25,
                    ),
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(
                                kernel_size=cnf.aug_gaussian_blur_kernel,
                                sigma=cnf.aug_gaussian_blur_sigma,
                            )
                        ],
                        p=0.25,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
                    # transforms.RandomPosterize(bits=4, p=0.1), # supported only for uint8 imgs
                    ConvertBCHWtoCBHW(),
                ]
            )
        else:
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    # ConvertBCHWtoCBHW(),
                ]
            )
            if cnf.noisy_bb_test:
                T_BB.transforms.append(NoisyBBIouPreserving(cnf.noisy_bb_iou_th))
            T_BB.transforms.append(Resize(cnf.input_h_w, antialias=None))
            T_CLIP = transforms.Compose(
                [
                    ConvertBCHWtoCBHW(),
                ]
            )
        return T_BB, T_CLIP

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for the MOTSynth dataset.

        Args:
            batch (list): List of tuples (image, targets, dists)

        Returns:
            tuple: (images, targets, dists)
        """
        clip, d_map, m_map, *other, clip_clean = zip(*batch)
        return (
            torch.stack(clip),
            torch.stack(d_map),
            torch.stack(m_map),
            *other,
            torch.stack(clip_clean),
        )


class MOTSynthInfer(VideoFrameDataset):
    def __init__(
        self,
        cnf: argparse.Namespace,
        detections_th=0.1,
        num_chunks=None,
        chunk_idx=None,
        sequence=None,
    ) -> None:
        super().__init__(cnf)
        self.FILENAME_LEN = 4
        self.video_path = self.cnf.ds_path
        self.MODE = "test"
        self.MAX_STRIDE = 8
        self.END_TO = 1800
        self.detections_th = detections_th
        self.infer_gt = cnf.infer_gt

        self.transform_bb, self.transform_clip = MOTSynth.get_transforms(
            self.cnf, self.MODE
        )

        if self.infer_gt:
            # normalization flow is trained on MOTSYNTH_TRAIN_STATIC_CLEAN_120, test: VAL-STATIC-5 + TEST-STATIC-10
            used_sequences = tuple(
                sorted(
                    MOTSYNTH_TRAIN_STATIC_CLEAN_120
                    + MOTSYNTH_VAL_STATIC_SEQUENCES[:5]
                    + MOTSYNTH_TEST_STATIC_SEQUENCES[:10]
                )
            )
        else:
            # inference on static sequences
            used_sequences = tuple(
                sorted(MOTSYNTH_VAL_STATIC_SEQUENCES + MOTSYNTH_TEST_STATIC_SEQUENCES)
            )

        self.name = f"MOTSynth_{'gt' if self.infer_gt else 'det'}"  # read externally (trainer_regressor.py)

        det_path = Path(self.cnf.detections_path)
        with Timer(f"Loading detections from {det_path}"):
            # get extension
            if det_path.suffix == ".npz":
                self.annotations = {
                    k: v for k, v in np.load(det_path).items() if k in used_sequences
                }
            elif det_path.is_dir():
                npys = list(det_path.glob("*.npy"))
                self.annotations = {
                    npy.stem[-3:]: np.load(npy)
                    for npy in npys
                    if npy.stem[-3:] in used_sequences
                }

        assert len(self.annotations) > 0, f"No detections found in {det_path}"
        assert (
            len(self.annotations) == len(used_sequences)
        ), f"Number of detections ({len(self.annotations)}) does not match number of sequences ({len(used_sequences)})"

        if num_chunks > 1:
            assert chunk_idx < num_chunks
            chunk_size = len(self.annotations) // num_chunks
            start_idx = chunk_idx * chunk_size
            end_idx = (
                (chunk_idx + 1) * chunk_size
                if chunk_idx < num_chunks - 1
                else len(self.annotations)
            )
            self.annotations = {
                k: v
                for k, v in self.annotations.items()
                if k in used_sequences[start_idx:end_idx]
            }

        if sequence is not None:
            # when sequence != None the inference is done on one sequence only
            assert sequence in self.annotations
            self.annotations = {sequence: self.annotations[sequence]}

        self.sequences = self.annotations.keys()

        with Timer(f"Loading videos from {self.video_path}"):
            self._load_videos()

    def get_frames_path(self, video: str) -> str:
        return f"frames/{video}/rgb"

    def __getitem__(self, idx: int):
        video_idx = np.searchsorted(self.cumulative_idxs, idx, side="right") - 1
        starting_frame = idx - self.cumulative_idxs[video_idx]
        clip, video_name, frames_name = self.videos[video_idx].get_clip(starting_frame)

        labels = self.annotations[video_name]
        detections_labels = [
            labels[labels[:, 0] == float(frame)][:, 1:] for frame in frames_name
        ]

        # x1, y1, x2, y2, (4) valid, (5) tracking_id, (6) dist, (7) visibility, x, y, z
        detections = [bboxes[:, 0:4] for bboxes in detections_labels]
        scores = [bboxes[:, 4] for bboxes in detections_labels]

        # filter out low confidence detections
        good_detections_idx = [np.where(s > self.detections_th)[0] for s in scores]
        detections = [
            detections[i][good_detections_idx[i]] for i in range(len(detections))
        ]
        scores = [scores[i][good_detections_idx[i]] for i in range(len(scores))]

        gt_dists = [
            np.zeros(len(good_detections_idx[i])) for i in range(len(detections_labels))
        ]
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

        # gt_dists = [torch.tensor(g, dtype=torch.float32) for g in gt_dists]
        detections = [torch.tensor(d, dtype=torch.float32) for d in detections]
        good_idxs = [
            torch.ones(len(detections[i]), dtype=torch.bool)
            for i in range(len(detections))
        ]
        clip, detections, _, good_idxs = self.transform_bb(
            clip, detections, gt_dists, good_idxs
        )
        assert all(
            torch.all(g == 1).item() for g in good_idxs
        ), "[CHECK] All detections should be valid during inference. INSERT 74-75-76"

        c_maps = torch.tensor([])
        for i, img in enumerate(clip):
            if len(detections[i]) == 0:
                c_maps = torch.cat((c_maps, torch.zeros((1, *self.input_size))), dim=0)
                continue

            if self.cnf.use_centers:
                c_map = gt.get_gt_cmap_only(
                    img=img, bboxes=detections[i], sigma=self.cnf.sigma
                )

                c_maps = torch.cat(
                    (c_maps, torch.tensor(c_map, dtype=torch.float).unsqueeze(0)), dim=0
                )

        clip = self.transform_clip(clip)

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


class MOTSynthSinglePath(MOTSynth):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf)
        self.FILENAME_LEN = 4
        self.MODE = mode
        self.min_visibility = self.cnf.min_visibility
        if self.cnf.use_debug_dataset:
            self.ANNOTATIONS_FILE = Path(
                "dataset/motsynth/motsynth_splits/motsynth_debug.txt"
            )
            self.MAX_STRIDE = 1
        else:
            self.ANNOTATIONS_FILE = Path(
                f"dataset/motsynth/motsynth_splits/motsynth_{mode}.txt"
            )
            self.MAX_STRIDE = self.cnf.max_stride

        with Timer(f"Loading {mode} annotations", one_line=True):
            self.sequences = tuple(self.ANNOTATIONS_FILE.read_text().splitlines())
            self.annotations = {
                seq: np.load(Path(self.annotations_path, f"{seq}.npy"))
                for seq in self.sequences
            }

        with Timer(f"Loading {mode} videos", one_line=True):
            self._load_videos()

    def __getitem__(self, idx: int):
        video_idx = np.searchsorted(self.cumulative_idxs, idx, side="right") - 1
        starting_frame = idx - self.cumulative_idxs[video_idx]
        video_name, frames_name = self.videos[video_idx].get_clip(
            starting_frame, only_labels=True
        )
        # select labels where the first element is equal to the frame name and valid
        labels = self.annotations[video_name]
        labels = [
            labels[
                (labels[:, 0] == float(frame))
                & (labels[:, 5] == 1.0)
                & (labels[:, 8] >= self.min_visibility)
            ][:, 1:]
            for frame in frames_name
        ]

        video_bboxes = [torch.from_numpy(bboxes[:, :4]) for bboxes in labels]
        # valid_flags = [torch.from_numpy(bboxes[:, 4]) for bboxes in labels]
        # tracking_ids = [torch.from_numpy(bboxes[:, 5]) for bboxes in labels]
        video_dists = [bboxes[:, 6] for bboxes in labels]
        visibilities = [torch.from_numpy(bboxes[:, 7]) for bboxes in labels]
        head_coords = [torch.from_numpy(bboxes[:, 8:11]) for bboxes in labels]
        # frame_id, x1, y1, x2, y2, valid_flag, track_id, distance, visibility, x, y, z
        assert len(head_coords[-1]) == len(video_bboxes[-1]) == len(video_dists[-1])

        path = Path(
            self.video_path, self.get_frames_path(video_name), f"{frames_name[0]}.jpg"
        )

        return path, video_bboxes, video_dists, visibilities

    @staticmethod
    def collate_fn(batch):
        return zip(*batch)

    def get_frames_path(self, video: str) -> str:
        return f"frames/{video}/rgb"
