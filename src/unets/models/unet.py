from typing import Sequence

import torch.nn as nn

from unets.models.modules import DownscaleBlock, ResidualBlock, UpscaleBlock


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: Sequence[int] = [64, 128, 256, 512, 1024],
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        self.input_block = ResidualBlock(in_channels, latent_channels[0], activation)

        self.encoder_blocks = nn.ModuleList(
            [
                DownscaleBlock(scale, next_scale, activation)
                for scale, next_scale in zip(latent_channels[:-1], latent_channels[1:])
            ]
        )

    def forward(self, x):
        features = [x]

        x = self.input_block(x)
        features.append(x)

        for module in self.encoder_blocks:
            x = module(x)
            features.append(x)

        return features


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: Sequence[int] = [64, 128, 256, 512, 1024],
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        self.bottleneck = ResidualBlock(
            latent_channels[-1], latent_channels[-1], activation
        )

        latent_channels = latent_channels[::-1]

        self.decoder_blocks = nn.ModuleList(
            [
                UpscaleBlock(scale, next_scale, activation)
                for scale, next_scale in zip(latent_channels[:-1], latent_channels[1:])
            ]
        )

        self.segmentation_head = ResidualBlock(
            latent_channels[-1], out_channels, nn.SiLU
        )

    def forward(self, features):
        features = features[1:]
        features = features[::-1]

        feature_map = self.bottleneck(features[0])

        for idx, decoder_block in enumerate(self.decoder_blocks):
            feature_map = decoder_block(feature_map, features[idx + 1])

        segmap = self.segmentation_head(feature_map)

        return segmap


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: Sequence[int] = [64, 128, 256, 512, 1024],
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        self.encoder = UNetEncoder(in_channels, latent_channels, activation)
        self.decoder = UNetDecoder(out_channels, latent_channels, activation)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
