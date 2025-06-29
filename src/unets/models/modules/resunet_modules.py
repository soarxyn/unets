import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: type[nn.Module]
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

        self.final_act = activation(inplace=True)

    def forward(self, x):
        return self.final_act(self.main_path(x) + self.shortcut(x))


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
