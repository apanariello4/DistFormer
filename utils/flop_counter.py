import logging
from functools import partial

import numpy as np
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset.kitti_ds import KITTI
from dataset.motsynth import MOTSynth
from dataset.nuscenes import NuScenes
from main import MODELS, parse_args
from utils.sampler import CustomSamplerTest
from utils.utils_scripts import is_list_empty


def flops_str_to_num(flops_str):
    fl, unit = flops_str.split()
    fl = float(fl)
    if unit == "M":
        fl *= 1e6
    elif unit == "G":
        fl *= 1e9
    elif unit == "T":
        fl *= 1e12
    return fl


def get_dataset(args):
    ds_test = {
        "motsynth": MOTSynth,
        "kitti": partial(KITTI, centers="zhu"),
        "kitti_3D": partial(KITTI, centers="kitti_3D"),
        "nuscenes": NuScenes,
    }.get(args.dataset)

    test_set = ds_test(args, mode="test")

    test_sampler = CustomSamplerTest(test_set, stride=None)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        pin_memory=True,
        worker_init_fn=test_set.wif_test,
        num_workers=args.n_workers,
        drop_last=True,
        collate_fn=test_set.collate_fn,
        sampler=test_sampler,
    )
    return test_loader


def warm_up_model(model, data, args):
    x, _, _, video_bboxes, *_ = data
    last_frame_bboxes = [v[-1] for v in video_bboxes]
    if "transformer" in args.regressor or args.regressor == "mae":
        bb = video_bboxes
    else:
        bb = last_frame_bboxes
    with torch.inference_mode():
        x = x.cuda()
        for _ in trange(int(50 / args.batch_size), desc="🔥 Warming up model 🔥"):
            model(x, bboxes=bb)


def flops(args, mode: str):
    model = MODELS[args.model](args)
    model = model.to("cuda")
    model.eval()
    test_loader = get_dataset(args)
    all_flops = []
    all_macs = []
    all_times = []
    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.ERROR)
    if mode == "timing":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        warm_up_model(model, next(iter(test_loader)), args)
    with tqdm(total=len(test_loader), desc=f"🚀 {mode} 🚀") as pbar:
        for data in test_loader:
            (
                x,
                _,
                _,
                video_bboxes,
                distances,
                visibilities,
                classes,
                _,
                frame_idx,
                video_idx,
                _,
                _,
                _,
            ) = data
            last_frame_bboxes = [v[-1] for v in video_bboxes]
            if is_list_empty(last_frame_bboxes):
                continue
            if "transformer" in args.regressor or args.regressor == "mae":
                bb = video_bboxes
            else:
                bb = last_frame_bboxes
            with torch.inference_mode():
                if mode == "timing":
                    x = x.cuda()
                    start.record()
                    _ = model(x, bboxes=bb)
                    end.record()
                    torch.cuda.synchronize()
                    all_times.append(start.elapsed_time(end))
                    pbar.set_postfix(
                        {"time": np.array(all_times).mean() / args.batch_size}
                    )

                elif mode == "flops":
                    flops, macs, params = get_model_profile(
                        model,
                        kwargs={"x": x.cuda(), "bboxes": bb},
                        print_profile=False,
                        detailed=False,
                        output_file="flops.txt",
                    )
                    all_flops.append(flops_str_to_num(flops))
                    all_macs.append(macs)
                    pbar.set_postfix(
                        {
                            "flops": flops,
                            "macs": macs,
                            "params": params,
                            "GFLOPs": np.array(all_flops).mean()
                            / args.batch_size
                            * 1e-9,
                        }
                    )

            pbar.update()

    if mode == "timing":
        print(f"Mean time: {np.array(all_times).mean() / args.batch_size:.2f} ms")
    elif mode == "flops":
        print(
            f"Mean flops: {np.array(all_flops).mean()/args.batch_size*1e-9:.2f} GFLOPs"
        )
        print(f"Mean macs: {np.array([float(x.split()[0]) for x in all_macs]).mean()}")
        print(f"Params: {params}")


if __name__ == "__main__":
    import sys

    common = [
        "--dataset",
        "kitti",
        "--ds_path",
        "data/datasets/kitti/",
        "--batch_size",
        "8",
        "--test_sampling_stride",
        "1",
        "--loss",
        "l1",
        "--n_workers",
        "4",
        "--input_h_w",
        "375",
        "1242",
    ]
    conf = {
        "distformer": [
            "--model",
            "distformer",
            "--regressor",
            "mae",
            "--backbone",
            "convnextfpn_base",
            "--pool_size",
            "8",
            "8",
            "--mae_version",
            "4",
            "--mae_crop_size",
            "120",
            "120",
            "--mae_encoder_layers",
            "6",
            "--mae_decoder_layers",
            "6",
        ],
        "distformer_mask": [
            "--model",
            "distformer",
            "--regressor",
            "mae",
            "--backbone",
            "convnextfpn_base",
            "--pool_size",
            "8",
            "8",
            "--mae_version",
            "4",
            "--mae_crop_size",
            "120",
            "120",
            "--mae_encoder_layers",
            "6",
            "--mae_decoder_layers",
            "6",
            "--mae_eval_masking",
            "--mae_eval_masking_ratio",
            "0.8",
        ],
        "zhu": [
            "--model",
            "zhu",
            "--fpn_idx_lstm",
            "none",
            "--pool_size",
            "2",
            "2",
            "--backbone",
            "convnextfpn_base",
            "--regressor",
            "roi_pooling",
        ],
    }
    sys.argv.extend(conf["distformer_mask"] + common)
    args = parse_args()

    print(
        f"\n{args.model=}, {args.batch_size=}, {args.mae_eval_masking=}, {args.mae_eval_masking_ratio=}\n"
    )
    flops(args, mode="timing")
