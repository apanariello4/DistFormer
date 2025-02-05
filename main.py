import argparse
import random
import socket
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.args import MODELS, parse_args, set_default_args
from utils.distributed import init_distributed


def set_seed(seed: Optional[int] = None) -> int:
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


def print_project_info(args: argparse.Namespace) -> None:
    HOSTNAME = socket.gethostname()
    project_name = str(Path.cwd().stem)
    m_str = f"┃ {project_name}@{HOSTNAME} ┃"
    u_str = "┏" + "━" * (len(m_str) - 2) + "┓"
    b_str = "┗" + "━" * (len(m_str) - 2) + "┛"
    print(u_str + "\n" + m_str + "\n" + b_str)
    print(f"\n{args}")
    if args.use_debug_dataset:
        print("\n", "\033[93m", "*" * 10, "Using debug dataset \033[0m")

    print(f"\n▶ Starting Experiment '{args.exp_name}' [seed: {args.seed}]")


def set_exp_name(args: argparse.Namespace):
    if args.exp_name is None and not args.resume:
        random_id = uuid.uuid4().hex[:6]
        if args.model in ("disnet", "mlp", "svr"):
            args.exp_name = f"{args.model}_{random_id}"
        else:
            args.exp_name = f"{args.model}_{args.backbone}_{args.regressor}_{args.input_h_w[0]}x{args.input_h_w[1]}_{random_id}"
    elif args.resume:
        args.exp_name = Path(args.checkpoint).parent.stem


def preprocess_args(args):
    assert Path(args.ds_path).exists(), f"Dataset path {args.ds_path} does not exist"

    if args.dataset == "motsynth":
        if args.annotations_path is None:
            args.annotations_path = Path(
                args.ds_path, "npy_annotations/annotations_clean"
            )
        else:
            args.annotations_path = Path(args.annotations_path)
        if not args.annotations_path.exists():
            raise ValueError("Annotations path is required for MOTSynth dataset")

    if args.debug:
        args.use_debug_dataset = True
        args.wandb = False

    if args.nearness:
        args.ds_stats["d_mean"] = -3.0915960980110233
        args.ds_stats["d_std"] = 0.7842383240216794

    args.seed = set_seed(args.seed)

    args = set_default_args(args)

    init_distributed(args)

    if args.device is None:
        args.device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"

    set_exp_name(args)
    print_project_info(args)

    if args.dataset == "motsynth":
        print("Dataset path: ", Path(args.ds_path).resolve())
        print("Annotations path: ", Path(args.annotations_path).resolve())

    if args.exp_log_path is None:
        if args.resume and args.checkpoint:
            args.exp_log_path = Path(args.checkpoint).parent
        else:
            args.exp_log_path = Path(args.log_path) / args.exp_name

    args.exp_log_path.mkdir(parents=True, exist_ok=True)
    return args


def main(args: argparse.Namespace) -> None:
    args = preprocess_args(args)
    model = MODELS[args.model](args)
    trainer = model.get_trainer()(model, args)

    trainer.run()


if __name__ == "__main__":
    main(args=parse_args())
