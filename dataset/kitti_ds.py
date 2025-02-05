import argparse
import multiprocessing
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Kitti
from tqdm import tqdm

from dataset.get_gt import gt_distances
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

MAX_FRAME_SIZE = (375, 1242)


class KITTI(Kitti):
    def __init__(
        self,
        cnf: argparse.Namespace,
        centers: str,
        mode: str = "train",
        segmentation=False,
    ) -> None:
        super(KITTI, self).__init__(cnf.ds_path, download=True)
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
        self.cnf = cnf
        self.clip_len = self.cnf.clip_len
        assert self.clip_len == 1, "clip_len must be 1"
        self.video_path = self.cnf.ds_path
        self.stride_sampling = self.cnf.stride_sampling
        self.input_size = MAX_FRAME_SIZE
        self.sampling = self.cnf.sampling
        self.mode = mode
        self.END_TO = None
        self.centers = centers
        self.classes_to_keep = self.get_classes_to_keep()
        self.segmentation = segmentation

        with Timer(f"Loading {mode} annotations", one_line=True):
            self._load_labels()

            self.annotations_file = Path(
                f'dataset/kitti/kitti_splits/{"train" if self.mode == "train" else "val"}.txt'
            )
            self.sequences = tuple(self.annotations_file.read_text().splitlines())

            self.labels = {k: v for k, v in self.labels.items() if k in self.sequences}

            # remove from labels elements with class DontCare or with distance < 0
            for k, v in self.labels.items():
                self.labels[k] = v[v[:, 0] != self.class_to_int["DontCare"]]

                self.labels[k] = self.labels[k][self.labels[k][:, -1] >= 0]
            self.images = [im for im in self.images if Path(im).stem in self.sequences]

        self.transform_bb, self.transform_clip = self.get_transforms()

        self.transforms = transforms.Compose(
            [
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                ConvertBCHWtoCBHW(),
            ]
        )

        self.frame_names = [Path(im).stem for im in self.images]
        if self.cnf.load_ds_into_ram:
            with Timer(f"Loading {mode} set into ram", one_line=True):
                self.load_images_parallel()

    def load_images_parallel(self):
        pool = multiprocessing.Pool()

        results = list(pool.map(self.load_image, self.images))
        pool.close()
        pool.join()

        self.images = results

    def load_image(self, image_path):
        return np.asarray(Image.open(image_path))[None, ...]

    def get_classes_to_keep(self):
        if self.cnf.classes_to_keep == "all":
            return list(self.class_to_int.values())

        return [self.class_to_int[c.capitalize()] for c in self.cnf.classes_to_keep]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        frame_name = self.frame_names[index]
        labels = self.labels[frame_name]

        classes, visibilities, video_bboxes, coords_3D, video_dists = self.get_labels(
            labels
        )

        if self.cnf.load_ds_into_ram:
            clip = self.images[index]
        else:
            clip = np.asarray(Image.open(self.images[index]))[None, ...]

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

        if self.segmentation:
            seg = np.asarray(
                Image.open(
                    f"/home/nello/TrackingLifter/data_object_instance_2/training/instance_2/{frame_name}.png"
                )
            )
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
                seg,
            )

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
        occlusions = torch.from_numpy(labels[:, 1])
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
        if self.cnf.annotations_path:
            labels_path = self.cnf.annotations_path
            print(f"Loading precomputed labels from {labels_path}")
        elif self.centers == "kitti_3D":
            labels_path = "data/kitti_labels_3DCenters.npz"
        else:
            labels_path = "data/kitti_labels_ZhuCenters.npz"

        if Path(labels_path).exists():
            self.labels = dict(np.load(labels_path))
            # remove last 4 columns (gt bbox) from labels if present
            for k, v in self.labels.items():
                if v.shape[1] == 14:
                    self.labels[k] = v[:, :-4]
                else:
                    break
        else:
            self.labels = {}
            for label in tqdm(
                self.targets,
                desc=f'Processing KITTI{"_3D" if "3D" in self.centers else "_Zhu"} labels',
            ):
                key = Path(label).stem
                label = self._parse_label(label)

                if self.centers == "kitti_3D":
                    label = np.concatenate(
                        (label, label[:, -1].reshape(len(label), 1)), axis=1
                    )
                    self.labels[key] = label
                else:
                    distances = gt_distances(
                        self.__class__.__name__, self.cnf.ds_path, key
                    )
                    assert len(distances) == label.shape[0]
                    distances = np.array(distances).reshape(len(distances), 1)
                    self.labels[key] = np.concatenate((label, distances), axis=1)

            Path("data").mkdir(exist_ok=True)
            np.savez(labels_path, **self.labels)
            print(f"Saved labels to {Path(labels_path).resolve()}")

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


class KITTIDisNet(KITTI):
    def __init__(self, cnf, mode="train"):
        super().__init__(cnf, mode)
        self.priors = {
            "Pedestrian": torch.tensor([1.75, 0.55, 0.30]),
            "Car": torch.tensor([1.53, 1.63, 3.88]),
            "Cyclist": torch.tensor([1.74, 0.60, 1.76]),
            "Tram": torch.tensor([3.53, 2.54, 16.09]),
            "Truck": torch.tensor([3.25, 2.59, 10.11]),
            "Van": torch.tensor([2.21, 1.90, 5.08]),
            "Misc": torch.tensor([1.91, 1.51, 3.58]),
            "Person_sitting": torch.tensor([1.27, 0.59, 0.80]),
            "DontCare": torch.tensor([-1.0, -1.0, -1.0]),
        }
        self.int_to_class = {v: k for k, v in self.class_to_int.items()}

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

        for label in labels:
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
            priors.append(self.priors[self.int_to_class[_class]])
            dists.append(dist)
            visibilities.append(visibility)
            classes.append(_class)

        priors = torch.stack(priors) if priors else torch.zeros((0, 3))
        x = torch.tensor([h_norm, w_norm, d_norm]).T
        x = torch.cat((x, priors), dim=1).type(torch.float32)
        dists = torch.tensor(dists).type(torch.float32)
        visibilities = torch.tensor(visibilities).type(torch.float32)
        classes = torch.tensor(classes).type(torch.float32)

        return x, dists, visibilities, classes

    @property
    def _raw_folder(self) -> str:
        return Path(f"{self.root}/KITTI/raw")


class KITTI_DIDM3D(Dataset):
    def __init__(self, args, mode="train") -> None:
        self.mode = mode
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
        self.cnf = args
        self.data_path = Path(self.cnf.ds_did3md).parent
        self.data = pd.read_feather(self.cnf.ds_did3md)
        if self.cnf.occluded_ds:
            self.data = self.data[self.data["gt_occluded"] == 2]

        self.unique_ids = self.data["img_id"].unique()

        self.transform_bb, self.transform_clip = self.get_transforms()

        self.transforms = transforms.Compose(
            [
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                ConvertBCHWtoCBHW(),
            ]
        )

    def __getitem__(self, idx):
        frame_idx = self.unique_ids[idx].replace(".txt", "")
        img_path = f"{self.data_path}/KITTI/raw/training/image_2/{frame_idx}.png"

        rows = self.data[self.data["img_id"] == self.unique_ids[idx]]

        clip = np.asarray(Image.open(img_path))[None, ...]
        video_bboxes = [
            torch.tensor(np.stack(rows["det_2d"].values)).type(torch.float32)
        ]
        video_dists = [rows["gt_z"].values]
        classes = [rows["class"].values]
        classes = [np.array([self.class_to_int[c] for c in classes[0]])]
        occlusions = [rows["gt_occluded"].values]
        visibilities = [torch.zeros(len(video_dists[0]))]
        coords_3D = [torch.zeros(len(video_dists[0]))]

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
            occlusions,
            classes,
            coords_3D,
            [frame_idx],
            0,
            tracking_ids,
            clip_clean,
        )

    def __len__(self):
        return len(self.data["img_id"].unique())

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
