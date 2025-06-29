import torch

from unets.models.modules import DownscaleBlock, ResidualBlock, UpscaleBlock


# --- Tests for Modules ---


def test_residual_block():
    res_block_same = ResidualBlock(16, 16, torch.nn.ReLU)
    x_same = torch.randn(4, 16, 64, 64)
    output_same = res_block_same(x_same)
    assert output_same.shape == (4, 16, 64, 64)

    res_block_diff = ResidualBlock(16, 32, torch.nn.ReLU)
    x_diff = torch.randn(4, 16, 64, 64)
    output_diff = res_block_diff(x_diff)
    assert output_diff.shape == (4, 32, 64, 64)


def test_down_block():
    in_channels, out_channels = 16, 32
    down_block = DownscaleBlock(in_channels, out_channels, torch.nn.ReLU)
    x = torch.randn(4, in_channels, 64, 64)
    output = down_block(x)
    assert output.shape == (4, out_channels, 32, 32)


def test_up_block():
    x1 = torch.randn(4, 64, 32, 32)
    x2 = torch.randn(4, 32, 64, 64)

    up_block = UpscaleBlock(64, 32, torch.nn.ReLU)
    output = up_block(x1, x2)

    assert output.shape == (4, 32, 64, 64)
