import argparse
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import Kitti

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

MAX_FRAME_SIZE = (900, 1600)


class NuScenes(Kitti):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf.ds_path, download=False)

        self.cnf = cnf
        self.clip_len = self.cnf.clip_len
        assert self.clip_len == 1, "clip_len must be 1"
        self.video_path = self.cnf.ds_path
        self.stride_sampling = self.cnf.stride_sampling
        self.input_size = self.cnf.input_h_w
        self.sampling = self.cnf.sampling
        self.mode = mode
        self.END_TO = None
        self.classes_to_keep = self.get_classes_to_keep()

        with Timer(f"Loading {mode} annotations", one_line=True):
            self.annotations_file = Path(
                f'dataset/nuscenes/nuscenes_splits/{"train" if self.mode == "train" else "val"}.txt'
            )

            self.sequences = tuple(self.annotations_file.read_text().splitlines())

            self._load_labels()

            self.labels = {k: v for k, v in self.labels.items() if k in self.sequences}

            # remove from labels elements with distance < 0
            for k in self.labels:
                self.labels[k] = self.labels[k][self.labels[k][:, -1] > 0]

            self.images = [
                im for im in self.images if Path(im).stem in self.labels
            ]

        self.transform_bb, self.transform_clip = self.get_transforms()

        self.transforms = transforms.Compose(
            [
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                ConvertBCHWtoCBHW(),
            ]
        )

        self.frame_names = [Path(im).stem for im in self.images]

    @staticmethod
    def load_image(image_path):
        return np.asarray(Image.open(image_path))[None, ...]

    @property
    def _raw_folder(self) -> str:
        return Path(self.root, "NuScenes", "raw")

    @property
    def class_to_int(self):
        return {
            "Car": 0,
            "Pedestrian": 1,
            "Truck": 2,
            "Trailer": 3,
            "Bus": 4,
            "Cons_vehicle": 5,
            "T_cone": 6,
            "Barrier": 7,
            "Motorcycle": 8,
            "Bicycle": 9,
        }

    def get_classes_to_keep(self):
        if self.cnf.classes_to_keep == "all":
            return list(self.class_to_int.values())
        return [self.class_to_int[c.lower()] for c in self.cnf.classes_to_keep]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        frame_name = self.frame_names[index]
        labels = self.labels[frame_name]

        classes, visibilities, video_bboxes, coords_3D, video_dists = self.get_labels(
            labels
        )

        clip = self.load_image(self.images[index])

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
            [frame_name],
            0,
            tracking_ids,
            None,
            clip_clean,
        )

    def get_labels(self, labels):
        labels = labels[np.isin(labels[:, 0], self.classes_to_keep)]
        classes = [labels[:, 0].astype(np.int8)]
        occlusions = torch.from_numpy(labels[:, 1].astype(np.float16))
        visibilities = [
            torch.ones(len(occlusions), dtype=torch.float16) - (occlusions / 3)
        ]
        video_bboxes = [torch.from_numpy(labels[:, 2:6].astype(np.float32))]
        coords_3D = [torch.from_numpy(labels[:, 9:12].astype(np.float32))]
        video_dists = [labels[:, -1].astype(np.float32)]
        return classes, visibilities, video_bboxes, coords_3D, video_dists

    def _parse_label(self, path: str) -> Dict[str, Any]:
        anns = pd.read_csv(path, sep=" ", header=None)
        anns[0] = anns[0].map(self.class_to_int)

        # select class_names(0), occlusion(2), bbox (4, 5, 6, 7), locations (11, 12, 13)
        label = anns.iloc[:, [0, 2, 4, 5, 6, 7, 11, 12, 13]]

        return label.to_numpy(dtype=np.float32)

    def _load_labels(self):
        labels_path = "data/nuscenes_labels.npz"

        if Path(labels_path).exists():
            self.labels = dict(np.load(labels_path))
        else:
            self.labels = {}
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(pool.imap(self.get_single_label, self.targets))

            for key, label in results:
                if len(label) > 0:
                    self.labels[key] = label

            Path("data").mkdir(exist_ok=True)
            np.savez(labels_path, **self.labels)

    def get_single_label(self, label: str):
        key = Path(label).stem
        try:
            label = self._parse_label(label)
        except pd.errors.EmptyDataError:
            # skip empty .txt files
            return (key, np.array([]))

        label = np.concatenate((label, label[:, -1].reshape(len(label), 1)), axis=1)

        return (key, label)

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

            T_BB.transforms.append(Resize(self.cnf.input_h_w, antialias=None))

        else:
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    Resize(self.cnf.input_h_w, antialias=None),
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


class NuScenesDisNet(NuScenes):
    def __init__(self, cnf, mode="train"):
        super().__init__(cnf, mode)
        import torch

        self.priors = {
            "Car": torch.tensor(
                [1.7323541866684955, 1.954289106432588, 4.62617036414758]
            ),
            "Pedestrian": torch.tensor(
                [1.7841543382646943, 0.6747440071590413, 0.7329673844747815]
            ),
            "Truck": torch.tensor(
                [2.8852555425904316, 2.525710618436406, 6.971358226371062]
            ),
            "Trailer": torch.tensor(
                [3.837754935552292, 2.911696851036058, 12.143243596018927]
            ),
            "Bus": torch.tensor(
                [3.561806148590948, 2.9514410760034155, 11.343868488471392]
            ),
            "Cons_vehicle": torch.tensor(
                [3.233891314895682, 2.862828723920427, 7.221200873362446]
            ),
            "T_cone": torch.tensor(
                [1.0944186585139892, 0.4377348526959422, 0.4431170094496943]
            ),
            "Barrier": torch.tensor(
                [0.9929638652050345, 2.5127619434294224, 0.49040113682501013]
            ),
            "Motorcycle": torch.tensor(
                [1.5090582770270269, 0.800012668918919, 2.120637668918919]
            ),
            "Bicycle": torch.tensor(
                [1.3259170854271356, 0.6238232830820771, 1.7382705192629815]
            ),
        }

    def __getitem__(self, idx):
        frame_name = Path(self.images[idx]).stem
        labels = self.labels[frame_name]
        h_norm = []
        w_norm = []
        d_norm = []
        visibilities = []
        classes = []
        dists = []
        priors = []

        for i, label in enumerate(labels):
            _class = label[0]
            visibility = label[1]
            bbox = label[2:6]
            dist = label[-1]
            x = float((float(bbox[2]) + float(bbox[0])) / 2.0) / float(
                MAX_FRAME_SIZE[1]
            )
            y = float((float(bbox[3]) + float(bbox[1])) / 2.0) / float(
                MAX_FRAME_SIZE[0]
            )
            width = float(float(bbox[2]) - float(bbox[0])) / float(MAX_FRAME_SIZE[1])
            height = float(float(bbox[3]) - float(bbox[1])) / float(MAX_FRAME_SIZE[0])
            H, W = MAX_FRAME_SIZE
            DIAG = np.sqrt(H**2 + W**2)
            h_norm.append(height / H)
            w_norm.append(width / W)
            d_norm.append(np.sqrt(height**2 + width**2) / DIAG)
            int_to_class = {v: k for k, v in self.class_to_int.items()}
            priors.append(self.priors[int_to_class[_class]])
            dists.append(dist)
            visibilities.append(visibility)
            classes.append(_class)

        priors = torch.stack(priors) if len(priors) > 0 else torch.zeros((0, 3))
        x = torch.tensor([h_norm, w_norm, d_norm]).T
        x = torch.cat((x, priors), dim=1).type(torch.float32)
        dists = torch.tensor(dists).type(torch.float32)
        visibilities = torch.tensor(visibilities).type(torch.float32)
        classes = torch.tensor(classes).type(torch.float32)

        return x, dists, visibilities, classes


class SVRNuscenesDataset(NuScenes):
    def __init__(self, cnf: argparse.Namespace, mode: str = "train") -> None:
        super().__init__(cnf, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor, Tensor]:
        frame_name = self.frame_names[idx]
        labels = self.labels[frame_name]

        classes, visibilities, video_bboxes, _, distances = self.get_labels(labels)

        hs = []
        ws = []
        if len(video_bboxes[0]) == 0:
            return (
                torch.zeros((0, 2)),
                torch.zeros((0,)),
                torch.zeros((0,)),
                torch.zeros((0,)),
            )
        for bbox in video_bboxes[0]:
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = x2 - x1
            hs.append(h)
            ws.append(w)

        x = torch.tensor([hs, ws]).T
        distances = torch.tensor(distances[0])
        visibilities = torch.tensor(visibilities[0])

        return x, distances, visibilities, torch.from_numpy(classes[0])
