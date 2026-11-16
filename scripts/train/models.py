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

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


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
