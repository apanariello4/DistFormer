from functools import partial

import torch

from models.temporal_compression.non_local import EfficientNonLocalBlock
from models.temporal_compression.base_module import TemporalCompression
from models.temporal_compression.lstm_module import LSTM
from models.temporal_compression.temporal_conv import TemporalConv


class LastFrame(TemporalCompression):
    def __init__(self, input_size: int, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, -1]

    @property
    def output_size(self):
        return self.input_size


class Identity(TemporalCompression):
    def __init__(self, input_size: int, **kwargs) -> None:
        super().__init__()
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def output_size(self):
        return self.input_size


TEMPORAL_COMPRESSORS = {
    "lstm": partial(LSTM, bidirectional=False),
    "bilstm": partial(LSTM, bidirectional=True),
    "conv": TemporalConv,
    "none": LastFrame,
    "identity": Identity,
    "non_local": EfficientNonLocalBlock,
}
