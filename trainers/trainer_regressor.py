import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.kitti_ds import KITTI, KITTI_DIDM3D
from dataset.kitti_tracking.kitti_tracking_ds import KittiTracking
from dataset.motchallenge import MOTChallengeInfer
from dataset.motsynth import MOTSynth, MOTSynthInfer
from dataset.nuscenes import NuScenes
from models.base_model import BaseLifter
from trainers.base_trainer import Trainer
from utils.bev import show_bev
from utils.metrics import get_metrics, print_metrics
from utils.plots import (
    average_variance_plot,
    bias_plot,
    distance_bins_plot,
    visibility_bins_plot,
)
from utils.sampler import (
    CustomDistributedSampler,
    CustomSamplerTest,
    CustomSamplerTrain,
)
from utils.utils_scripts import is_list_empty, nearness_to_z, z_to_nearness

plt.switch_backend("agg")


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class TrainerRegressor(Trainer):
    def __init__(self, model: BaseLifter, args: argparse.Namespace) -> None:
        super().__init__(model, args)

    def train(self) -> None:
        self.model.train()

        model = self.model.module if self.cnf.distributed else self.model

        losses = defaultdict(list)

        scaler = torch.amp.GradScaler(init_scale=(2**16) / self.cnf.accumulation_steps, enabled=self.cnf.fp16)
        self.optimizer.zero_grad()
        tot_bboxes = 0
        tot_acc = 0.0
        loss_fun = model.get_loss_fun()

        barrier()

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.epoch)

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}/{self.cnf.epochs}",
        ) as pbar:
            for step, sample in enumerate(self.train_loader):
                assert (len(self.train_loader) / self.cnf.batch_size > self.cnf.accumulation_steps)

                (x, _, _, video_bboxes, distances, _, _, head_coords, frame_idx, video_idx, _, gt_anchors, clip_clean) = sample
                last_frame_bboxes = [v[-1] for v in video_bboxes]

                if any(map(is_list_empty, (last_frame_bboxes, video_bboxes, distances))):
                    pbar.update()
                    continue

                x = x.to(self.cnf.device)

                y_true = torch.cat([d[-1] for d in distances]).to(self.cnf.device)
                y_true = z_to_nearness(y_true) if self.cnf.nearness else y_true

                with torch.autocast(
                    self.cnf.device.split(":")[0],
                    enabled=self.cnf.fp16,
                    dtype=torch.float16,
                ):
                    if ("transformer" in self.cnf.regressor or "mae" in self.cnf.regressor):
                        save_crops = (step % self.cnf.save_freq == 0 or step == len(self.train_loader) - 1)
                        do_mae = (self.epoch < self.cnf.disable_mae_epoch or self.cnf.disable_mae_epoch < 0)
                        output = self.model(
                            x,
                            video_bboxes,
                            clip_clean=clip_clean,
                            save_crops=save_crops,
                            gt_anchors=gt_anchors,
                            dataset_anchor=None,
                            do_mae=do_mae,
                        )
                    else:
                        output = self.model(x, last_frame_bboxes)

                    if "mae" in self.cnf.regressor:
                        mae_losses = output[-1]
                        for k, v in mae_losses.items():
                            losses[k].append(v.item())
                        output = output[:-1] if len(output) == 3 else output[0]

                    loss_warmup = self.get_warmup(warmup_start=self.cnf.loss_warmup_start, warmup_end=self.cnf.loss_warmup)
                    loss = loss_fun(y_pred=output, y_true=y_true, bboxes=last_frame_bboxes)
                    losses["distance_loss"].append(loss.item())
                    loss = loss * loss_warmup
                    if self.cnf.regressor == "mae" and do_mae:
                        mae_warmup = self.get_warmup(warmup_start=0, warmup_end=self.cnf.mae_warmup)
                        tot_mae_loss = sum(mae_losses.values())
                        loss += self.cnf.mae_alpha * tot_mae_loss * mae_warmup

                        if self.cnf.mae_loss_only:
                            loss = tot_mae_loss

                    loss = loss / self.cnf.accumulation_steps

                scaler.scale(loss).backward()
                losses["loss"].append(loss.item())

                if isinstance(output, tuple) and len(output) == 2:
                    output = output[0]
                output = output.detach().cpu().numpy().tolist()

                y_pred_z = (nearness_to_z(torch.tensor(output)).numpy() if self.cnf.nearness else output)

                errors = np.abs(y_pred_z - y_true.cpu().detach().numpy())
                hits = sum(errors < 1)
                tot_bboxes += len(y_true)
                tot_acc += hits
                epoch_acc = tot_acc / tot_bboxes

                if (step + 1) % self.cnf.accumulation_steps == 0:
                    if self.cnf.grad_clip:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_value_(model.parameters(), self.cnf.grad_clip)
                    scaler.step(self.optimizer)
                    self.optimizer.zero_grad()
                    scaler.update()
                    if self.cnf.scheduler == "cosine":
                        self.scheduler.step()

                if (
                    self.epoch % self.cnf.lr_decay_steps == 0
                    and self.epoch > 0
                    and self.cnf.scheduler != "plateau"
                ):
                    self.scheduler.base_lrs[0] *= self.cnf.lr_gamma

                pbar.set_postfix(
                    {
                        "alp@1": epoch_acc,
                        "loss": np.mean(losses["loss"]),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                pbar.update()
                if wandb.run:
                    wandb.log(
                        {"it_train_acc": hits / len(y_true)}
                        | {k: v[-1] for k, v in losses.items()}
                    )

                barrier()

        if wandb.run:
            wandb.log(
                {
                    "epoch": self.epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_train_acc": epoch_acc,
                    "train_it": (step + 1) * (self.epoch + 1),
                }
                | {f"epoch_train_{k}": np.mean(v) for k, v in losses.items()}
            )

    def test(self) -> None:
        if self.cnf.loss_warmup_start > self.epoch and not self.cnf.test_only:
            print(
                f'{"":<5}│ Skipping test for epoch {self.epoch + 1} since distance loss has not been introduced yet'
            )
            return
        self.model.eval()

        model = self.model.module if self.cnf.distributed else self.model

        t = time()
        all_errors = np.array([], dtype=np.float32)
        all_true = np.array([], dtype=np.float32)
        all_vars = np.array([], dtype=np.float32)
        all_pred = np.array([], dtype=np.float32)
        all_visibilities = np.array([], dtype=np.float32)
        all_idx = np.array([], dtype=np.int32)
        all_video_idx = np.array([], dtype=np.int32)
        all_classes = np.array([], dtype=np.int32)
        hits = 0
        tot = 0
        tot_loss = 0.0
        loss_fun = model.get_loss_fun()

        barrier()
        with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
            for step, sample in enumerate(self.test_loader):
                (x, _, _, video_bboxes, distances, visibilities, classes, _, frame_idx, video_idx, _, _, clip_clean) = sample

                last_frame_visibilities = [v[-1] for v in visibilities]
                last_frame_bboxes = [v[-1] for v in video_bboxes]
                last_frame_classes = [c[-1] for c in classes]

                if is_list_empty(last_frame_bboxes):
                    pbar.update()
                    continue

                x = x.to(self.cnf.device)

                if "transformer" in self.cnf.regressor or self.cnf.regressor == "mae":
                    output = self.model(x, video_bboxes)
                else:
                    output = self.model(x, last_frame_bboxes)

                y_true = torch.cat([d[-1] for d in distances]).to(self.cnf.device)
                y_true = z_to_nearness(y_true) if self.cnf.nearness else y_true
                loss = loss_fun(y_pred=output, y_true=y_true, bboxes=last_frame_bboxes)
                dists_pred, dists_vars = self.post_process(output)

                if self.cnf.show_bev:
                    show_bev(
                        self.cnf.dataset,
                        dists_pred,
                        dists_vars,
                        video_bboxes,
                        clip_clean,
                        y_true,
                        self.cnf.bev_out_path,
                        video_idx[-1],
                        frame_idx[-1][0],
                    )

                tot_loss += loss.item()
                all_vars = np.append(all_vars, dists_vars)
                all_visibilities = np.append(
                    all_visibilities, np.concatenate(last_frame_visibilities)
                )
                all_classes = np.append(all_classes, np.concatenate(last_frame_classes))
                dists_pred = (
                    nearness_to_z(np.asarray(dists_pred)).tolist()
                    if self.cnf.nearness
                    else dists_pred
                )

                all_pred = np.append(all_pred, dists_pred)
                y_true = y_true.cpu().detach().numpy()
                errors = np.abs(y_true - np.array(dists_pred))
                all_errors = np.append(all_errors, errors)
                all_true = np.append(all_true, y_true)
                hits += len(errors[errors < 1])
                tot += len(errors)

                pbar.update()
                pbar.set_postfix({"alp@1": hits / tot, "loss": tot_loss / (step + 1)})

        all_results = {
            "all_errors": all_errors,
            "all_true": all_true,
            "all_vars": all_vars,
            "all_pred": all_pred,
            "all_visibilities": all_visibilities,
            "all_classes": all_classes,
            "all_idx": all_idx,
            "all_video_idx": all_video_idx,
        }
        self.save_results(
            filename=f"epoch_{self.epoch}_test_results.npz", results=all_results
        )

        acc = hits / tot
        current_error_mean = np.mean(all_errors)
        if self.cnf.scheduler == "plateau":
            self.scheduler.step(current_error_mean)

        error_std = np.std(all_errors)

        metrics = get_metrics(
            y_true=all_true,
            y_pred=all_pred,
            classes_mapping=self.test_loader.dataset.class_to_int,
            y_visibilities=all_visibilities,
            y_classes=all_classes,
            long_range=self.cnf.long_range,
            max_dist=self.cnf.max_dist_test,
        )
        current_rmse = metrics["all"]["rmse_linear"]

        if wandb.run:
            wandb.log(
                {
                    "acc": acc,
                    "error_mean": current_error_mean,
                    "error_std": error_std,
                    "test_loss": tot_loss / len(self.test_loader),
                    "epoch": self.epoch,
                    "test_it": (step + 1) * (self.epoch + 1),
                }
                | metrics["all"]
                | metrics
            )
            bias_plot(pred_dists=all_pred, true_dists=all_true)
            distance_bins_plot(all_errors=all_errors, all_true=all_true, acc=acc)
            visibility_bins_plot(
                all_visibilities=all_visibilities, all_errors=all_errors, acc=acc
            )
            if len(all_vars) > 0:
                average_variance_plot(all_vars, all_true)
            plt.close("all")

        # save best model
        if (
            self.best_test_rmse_linear is None
            or current_rmse < self.best_test_rmse_linear
        ):
            self.best_test_rmse_linear = current_rmse
            if not self.cnf.save_nothing:
                if self.cnf.distributed and self.cnf.rank == 0:
                    self.model.module.save_w(self.log_path / "best.pth")
                else:
                    self.model.save_w(self.log_path / "best.pth", self.cnf)
            self.patience = self.cnf.max_patience
            self.save_results(
                filename="best_rmse_test_results.npz", results=all_results
            )
        elif torch.isnan(loss).any():
            print("NaN loss, exiting")
            self.patience = 0
        else:
            self.patience -= 1 if self.epoch >= self.cnf.loss_warmup_start else 0

        if wandb.run:
            wandb.log(
                {
                    "patience": self.patience,
                    "epoch": self.epoch,
                    "best_rmse": self.best_test_rmse_linear,
                }
            )

        if self.cnf.rank == 0:
            print(
                f"\r \t● (ACC) on TEST-set: "
                f"({acc:.2%}) "
                f"│ P: {self.patience} / {self.cnf.max_patience} "
                f"| Error Mean: {current_error_mean.item():.2f} ± {error_std.item():.2f} "
                f"| RMSE Lin: {current_rmse.item():.2f} "
                f"| Loss: {tot_loss / len(self.test_loader):.4f} "
                f"| T: {time() - t:.2f} s "
            )

            print_metrics(metrics)

        barrier()

    def infer(self) -> None:
        print(f'{"":<5}│ Starting inference ... will be saved at {self.log_path}')
        self.model.eval()
        last_sequence = None
        detections_to_save = []
        frame_features_list = []
        frame_features_to_save = {}
        is_mot_challenge = isinstance(self.test_loader.dataset, MOTChallengeInfer)
        save_path = (
            self.log_path / "motchallenge"
            if is_mot_challenge
            else self.log_path / "motsynth"
        )

        def hook_fn(m, i, o):
            setattr(m, "regressor_input", i)

        _ = self.model.regressor.register_forward_hook(hook_fn)

        for sample in self.test_loader:
            if self.cnf.infer_gt:
                (
                    x,
                    video_bboxes,
                    detections_raw,
                    scores,
                    track_ids,
                    gt_dists,
                    vis,
                    video_name,
                    frame_name,
                ) = sample
            else:
                x, video_bboxes, detections_raw, scores, video_name, frame_name = sample

            last_frame_bboxes = [v[-1] for v in video_bboxes]
            last_frame_raw_detections = [v[-1] for v in detections_raw]
            last_frame_scores = [v[-1] for v in scores]
            last_frame_track_ids = (
                [v[-1] for v in track_ids] if self.cnf.infer_gt else None
            )
            last_frame_gt_distances = (
                [v[-1] for v in gt_dists] if self.cnf.infer_gt else None
            )
            last_frame_visibilities = (
                [v[-1] for v in vis] if self.cnf.infer_gt else None
            )

            if is_list_empty(last_frame_bboxes):
                x = x.to(self.cnf.device)
                fake_bbox = [0, 0, self.cnf.input_h_w[0] - 1, self.cnf.input_h_w[1] - 1]
                fake_input = [
                    torch.tensor(fake_bbox, dtype=torch.float32).reshape(1, 4)
                    for _ in range(self.cnf.batch_size)
                ]
                if "transformer" in self.cnf.regressor or self.cnf.regressor == "mae":
                    assert (
                        self.cnf.clip_len == 1
                    ), "Transformer infer only works with clip_len=1 (temporary limitation)"
                    self.model(x, [[v] for v in fake_input])
                else:
                    self.model(x, fake_input)
                features = self.model.regressor.regressor_input[0]
                features = features if features.shape == 4 else features.squeeze(2)
                features = nn.AvgPool2d(
                    kernel_size=(features.shape[-2], features.shape[-1])
                )(features)
                # all frames are empty, save empty detections and continue
                for i, (curr_video, curr_frame) in enumerate(
                    zip(video_name, frame_name)
                ):
                    if curr_video != last_sequence and last_sequence is not None:
                        self.save_detections(
                            detections_to_save,
                            save_path,
                            last_sequence,
                            self.cnf.infer_gt,
                        )
                        frame_features_to_save[last_sequence] = np.array(
                            frame_features_list, dtype=np.float32
                        )
                        detections_to_save = []
                        frame_features_list = []
                    frame_features_list.append(features[i].cpu().numpy().reshape(-1))
                    last_sequence = curr_video
                continue

            x = x.to(self.cnf.device)
            if "transformer" in self.cnf.regressor or self.cnf.regressor == "mae":
                output = self.model(x, video_bboxes)
            else:
                output = self.model(x, last_frame_bboxes)

            # get frame features from the model using the result from the hook
            features = self.model.regressor.regressor_input[0]
            features = features if features.shape == 4 else features.squeeze(2)
            features = nn.AvgPool2d(
                kernel_size=(features.shape[-2], features.shape[-1])
            )(features)

            # features = self.model.regressor.featss
            dists_pred, dists_logvars = self.post_process(output)

            dists_pred = nearness_to_z(dists_pred) if self.cnf.nearness else dists_pred

            # given flatten detections over batches, we need to re-group them
            cum_sum = np.cumsum([v.shape[0] for v in last_frame_bboxes])[:-1]
            dists_pred = np.split(dists_pred, cum_sum)
            dists_vars = (
                np.split(np.exp(dists_logvars), cum_sum)
                if len(dists_logvars) > 0
                else None
            )

            # save detections
            for i, (curr_video, curr_frame) in enumerate(zip(video_name, frame_name)):
                if curr_video != last_sequence and last_sequence is not None:
                    self.save_detections(
                        detections_to_save, save_path, last_sequence, self.cnf.infer_gt
                    )
                    frame_features_to_save[last_sequence] = np.array(
                        frame_features_list, dtype=np.float32
                    )
                    detections_to_save = []
                    frame_features_list = []
                for j in range(len(last_frame_raw_detections[i])):
                    detection = last_frame_raw_detections[i][j]
                    score = last_frame_scores[i][j]
                    distance = dists_pred[i][j]
                    variance = dists_vars[i][j] if dists_vars is not None else -1
                    if self.cnf.infer_gt:
                        track_id = last_frame_track_ids[i][j]
                        gt_distance = last_frame_gt_distances[i][j]
                        visibility = last_frame_visibilities[i][j]
                        array_input = np.array(
                            (
                                int(curr_frame),
                                *detection,
                                score,
                                distance,
                                variance,
                                track_id,
                                gt_distance,
                                visibility,
                            ),
                            dtype=np.float32,
                        )
                    else:
                        array_input = np.array(
                            (int(curr_frame), *detection, score, distance, variance),
                            dtype=np.float32,
                        )

                    detections_to_save.append(array_input)

                # save features
                frame_features_list.append(features[i].cpu().numpy().reshape(-1))

                last_sequence = curr_video

            print(f"\r{self.test_progress_bar} " f"| curr_vid: {curr_video} ", end="")

            self.test_progress_bar.inc()

        self.save_detections(
            detections_to_save, save_path, last_sequence, self.cnf.infer_gt
        )

        frame_features_to_save[last_sequence] = np.array(
            frame_features_list, dtype=np.float32
        )
        if self.cnf.infer_frame_features:
            self.save_frame_features(
                frame_features_to_save,
                save_path,
                self.test_loader.dataset.name,
                self.cnf.infer_chunk_idx,
                self.cnf.infer_sequence,
            )
        print(f'\r \t● inference done\n {"":<5}│ Results saved in {save_path}')

    def post_process(self, output):
        if isinstance(output, tuple):
            # with variance
            dists_pred = output[0].detach().cpu().numpy()
            dists_vars = output[1].detach().cpu().numpy()
        else:
            # without variance
            dists_pred = output.detach().cpu().numpy()
            dists_vars = []
        return dists_pred, dists_vars

    @staticmethod
    def save_detections(detections_to_save, log_path, last_sequence, is_gt):
        sequences_dir_name = "gt_sequences" if is_gt else "sequences"
        sequences_path = Path(log_path, sequences_dir_name)
        sequences_path.mkdir(exist_ok=True, parents=True)
        np.save(sequences_path / f"{last_sequence}.npy", detections_to_save)

    @staticmethod
    def save_frame_features(
        frame_features: dict, log_path, dataset_name, chunk_idx, infer_sequence
    ):
        chunk_idx = f"{chunk_idx}_" if chunk_idx is not None else ""
        infer_sequence = f"_{infer_sequence}" if infer_sequence is not None else ""
        path = (
            Path(log_path)
            / f"{infer_sequence}{chunk_idx}{dataset_name}_frame_features_sequences.npz"
        )
        np.savez(path, **frame_features)
        print(f"frame features saved in {path}")

    def get_warmup(self, warmup_start: int = 0, warmup_end: int = 0) -> float:
        lin_space = np.zeros(self.cnf.epochs, dtype=np.float32)
        lin_space[warmup_start:warmup_end] = np.linspace(
            0, 1, warmup_end - warmup_start
        )
        lin_space[warmup_end:] = 1
        return lin_space[self.epoch]

    def get_dataset(self, args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
        ds_train = {
            "motsynth": MOTSynth,
            "kitti": partial(KITTI, centers="zhu"),
            "kitti_3D": partial(KITTI, centers="kitti_3D"),
            "kitti_tracking": KITTI,  # kitti tracking does not have train set
            "kitti_didm3d": partial(KITTI, centers="kitti_3D"),
            "nuscenes": NuScenes,
        }.get(self.cnf.dataset)

        ds_test = {
            "motsynth": MOTSynth,
            "kitti": partial(KITTI, centers="zhu"),
            "kitti_3D": partial(KITTI, centers="kitti_3D"),
            "kitti_tracking": KittiTracking,
            "kitti_didm3d": KITTI_DIDM3D,
            "nuscenes": NuScenes,
        }.get(self.cnf.dataset)

        if self.cnf.dataset == "kitti_didm3d":
            assert Path(
                self.cnf.ds_did3md
            ).exists(), "ds_did3md must be specified for kitti_didm3d eval dataset"
        training_set = None
        test_set = ds_test(args, mode="test")

        if not self.cnf.test_only and not self.cnf.infer_only:
            training_set = ds_train(args)
            if self.cnf.distributed:
                train_sampler = CustomDistributedSampler(
                    training_set, stride=args.train_sampling_stride
                )
            else:
                train_sampler = CustomSamplerTrain(
                    training_set, self.cnf.seed, stride=args.train_sampling_stride
                )
            self.train_loader = DataLoader(
                dataset=training_set,
                batch_size=args.batch_size,
                num_workers=args.n_workers,
                pin_memory=True,
                worker_init_fn=training_set.wif,
                drop_last=True,
                collate_fn=training_set.collate_fn,
                sampler=train_sampler,
            )

        if self.cnf.infer_only:
            if self.cnf.infer_mot_version:
                assert self.cnf.infer_split, "You must specify the split to infer"
                test_set = MOTChallengeInfer(args)
            else:
                test_set = MOTSynthInfer(
                    args,
                    num_chunks=self.cnf.infer_num_chunks,
                    chunk_idx=self.cnf.infer_chunk_idx,
                    sequence=self.cnf.infer_sequence,
                )
            infer_sampler = (
                CustomSamplerTest(test_set)
                if not self.cnf.distributed
                else CustomDistributedSampler(test_set, train=False)
            )
            self.test_loader = DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                pin_memory=True,
                worker_init_fn=test_set.wif_test,
                num_workers=args.n_workers,
                drop_last=False,
                collate_fn=test_set.collate_fn,
                sampler=infer_sampler,
            )
        else:
            stride = (
                None
                if args.test_all or self.cnf.dataset != "motsynth"
                else args.test_sampling_stride
            )
            if not self.cnf.distributed:
                test_sampler = CustomSamplerTest(test_set, stride=stride)
            else:
                test_sampler = CustomDistributedSampler(
                    test_set, train=False, stride=stride
                )
            self.test_loader = DataLoader(
                dataset=test_set,
                batch_size=args.batch_size,
                pin_memory=True,
                worker_init_fn=test_set.wif_test,
                num_workers=args.n_workers,
                drop_last=True,
                collate_fn=test_set.collate_fn,
                sampler=test_sampler,
            )
        return training_set, test_set
