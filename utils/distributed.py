import argparse
import os

import torch


def init_distributed(args: argparse.Namespace) -> None:
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        torch.cuda.set_device(f"cuda:{args.rank}")

    assert args.rank >= 0
