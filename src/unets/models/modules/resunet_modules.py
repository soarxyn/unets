import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
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

    def forward(self, x: Float[Tensor, "N Cin H W"]) -> Float[Tensor, "N Cout H W"]:
        return self.final_act(self.main_path(x) + self.shortcut(x))


@jaxtyped
class DownscaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module):
        super().__init__()

        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2), ResidualBlock(in_channels, out_channels, activation)
        )

    def forward(self, x: Float[Tensor, "N C H W"]) -> Float[Tensor, "N C H/2 W/2"]:
        return self.mp_conv(x)


@jaxtyped
class UpscaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_mode: str,
        activation: nn.Module,
    ):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode=upscale_mode, align_corners=True
        )
        self.conv = nn.Sequential(
            ResidualBlock(in_channels + out_channels, out_channels, activation),
            ResidualBlock(out_channels, out_channels, activation),
        )

    def forward(
        self,
        feature_map: Float[Tensor, "B Cin H/2 W/2"],
        skip_connection: Float[Tensor, "B Cout H W"],
    ) -> Float[Tensor, "B Cout H W"]:
        feature_map = self.upsample(feature_map)
        feature_map = torch.cat([feature_map, skip_connection], dim=1)
        feature_map = self.conv(feature_map)
        return feature_map
