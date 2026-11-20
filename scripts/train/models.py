import math

import torch
import torch.nn.functional as F
from torch import nn


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class GestureCNN(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=3,
        channels=64,
        kernel_size=5,
        dilations=(1, 2, 4, 8, 16),
        dropout=0.1,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.head = nn.Conv1d(channels, num_classes, 1)

        self.receptive_field = 1 + (kernel_size - 1) * (1 + 2 * sum(dilations))
        self.context = self.receptive_field
        self.inference_chunk = None

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class GestureCNNGRU(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=3,
        channels=64,
        hidden=64,
        kernel_size=5,
        dilations=(1, 2),
        dropout=0.1,
        bidirectional=True,
        context=240,
        inference_chunk=480,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.gru = nn.GRU(
            channels,
            hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(hidden * (2 if bidirectional else 1), num_classes, 1)

        self.context = context
        self.inference_chunk = inference_chunk

    def forward(self, x):
        features = self.blocks(self.stem(x)).transpose(1, 2)
        sequence, _ = self.gru(features)
        return self.head(self.dropout(sequence).transpose(1, 2))


def sinusoidal_positions(length, dim, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table


class GestureTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=3,
        channels=64,
        heads=4,
        layers=3,
        patch=4,
        kernel_size=5,
        dropout=0.1,
        context=240,
        inference_chunk=480,
    ):
        super().__init__()

        self.patch = patch

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.down = nn.Conv1d(channels, channels, patch, stride=patch)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                channels,
                heads,
                dim_feedforward=channels * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            layers,
            enable_nested_tensor=False,
        )
        self.up = nn.ConvTranspose1d(channels, channels, patch, stride=patch)
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(channels, num_classes, 1)

        self.context = context
        self.inference_chunk = inference_chunk

    def forward(self, x):
        length = x.shape[-1]
        x = F.pad(x, (0, (-length) % self.patch))

        local = self.stem(x)
        tokens = self.down(local).transpose(1, 2)
        tokens = tokens + sinusoidal_positions(tokens.shape[1], tokens.shape[2], x.device)
        tokens = self.encoder(tokens).transpose(1, 2)

        merged = self.refine(self.up(tokens) + local)
        return self.head(merged)[..., :length]


GESTURE_MODELS = {
    "cnn": GestureCNN,
    "cnn_gru": GestureCNNGRU,
    "transformer": GestureTransformer,
}


def build_gesture_model(name, **kwargs):
    return GESTURE_MODELS[name](**kwargs)


class EpisodeCNN(nn.Module):
    def __init__(
        self,
        in_channels=6,
        widths=(32, 64, 64, 128),
        kernel_sizes=(7, 5, 5, 3),
        pool=4,
        dropout=0.3,
    ):
        super().__init__()

        layers = []
        previous = in_channels

        for index, (width, kernel_size) in enumerate(zip(widths, kernel_sizes)):
            layers += [
                nn.Conv1d(previous, width, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            ]

            if index < len(widths) - 1:
                layers.append(nn.MaxPool1d(pool))

            previous = width

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(previous, 1),
        )

    def forward(self, x):
        return self.head(self.features(x).mean(dim=2)).squeeze(1)
