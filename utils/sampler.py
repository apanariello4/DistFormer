import math
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class CustomSamplerBase(Sampler):
    def __init__(self, dataset: Dataset, stride: Optional[int] = None) -> None:
        self.dataset = dataset
        self.num_samples = stride
        self.len_data = len(dataset)
        self.size = self.len_data // stride if stride is not None else self.len_data

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.size


class CustomSamplerTrain(CustomSamplerBase):
    def __init__(
        self, dataset: Dataset, seed: int, stride: Optional[int] = None
    ) -> None:
        self.rng = np.random.default_rng(seed)
        super().__init__(dataset, stride)

    def __iter__(self) -> Iterator[int]:
        return iter(self.rng.choice(self.len_data, size=self.size, replace=False))


class CustomSamplerTest(CustomSamplerBase):
    def __init__(self, dataset: Dataset, stride: Optional[int] = None) -> None:
        super().__init__(dataset, stride)

    def __iter__(self) -> Iterator[int]:
        return iter(np.arange(0, self.len_data, self.num_samples))


class CustomDistributedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        stride: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        train: bool = True,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(
                    rank, num_replicas - 1
                )
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.stride = stride
        self.train = train
        self.num_samples = len(self.dataset) // (self.stride * self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if self.train:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[
                :: self.stride
            ].tolist()
        else:
            indices = torch.arange(0, len(self.dataset), self.stride)[
                :: self.stride
            ].tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
