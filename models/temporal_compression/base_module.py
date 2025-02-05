import torch.nn as nn


class TemporalCompression(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def output_size(self):
        raise NotImplementedError
