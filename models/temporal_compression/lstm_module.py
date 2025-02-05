from models.temporal_compression.base_module import TemporalCompression
from models.temporal_compression.biconvlstm import ConvLSTM


class LSTM(TemporalCompression):
    def __init__(
        self,
        input_size,
        backbone,
        hidden_dim: int = 256,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.frame_size = backbone.frame_size
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = ConvLSTM(
            input_dim=self.input_size,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            img_size=self.frame_size,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        x = self.lstm(x)[0].squeeze(1)
        return x

    @property
    def output_size(self):
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
