import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()

        squeezed_channels = max(1, channels // reduction_ratio)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, squeezed_channels, bias=False),
            nn.SiLU(),
            nn.Linear(squeezed_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: type[nn.Module],
        use_se: bool = False,
    ):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.se_block = SEBlock(out_channels) if use_se else nn.Identity()

        self.final_act = activation(inplace=True)

    def forward(self, x):
        main_out = self.main_path(x)
        main_out = self.se_block(main_out)

        return self.final_act(main_out + self.shortcut(x))


class DownscaleBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: type[nn.Module]
    ):
        super().__init__()

        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2), ResidualBlock(in_channels, out_channels, activation)
        )

    def forward(self, x):
        return self.mp_conv(x)


class UpscaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: type[nn.Module],
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ResidualBlock(in_channels + out_channels, out_channels, activation),
            ResidualBlock(out_channels, out_channels, activation),
        )

    def forward(
        self,
        feature_map,
        skip_connection,
    ):
        feature_map = self.upsample(feature_map)
        feature_map = torch.cat([feature_map, skip_connection], dim=1)
        feature_map = self.conv(feature_map)
        return feature_map
