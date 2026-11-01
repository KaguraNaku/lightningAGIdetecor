"""Fixed residual filters and checkpoint-compatible unused Gaussian modules."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def residual_kernels():
    filters = np.asarray(
        [
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=float,
    )
    for index, divisor in enumerate((4.0, 12.0, 2.0)):
        filters[index] /= divisor
    return filters[:, None]


class ResidualFilter(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        self.kernel = nn.Parameter(
            torch.tensor(
                np.repeat(residual_kernels(), in_channels, axis=1), dtype=torch.float32
            ),
            requires_grad=False,
        )

    def forward(self, images):
        return self.truc(F.conv2d(images, self.kernel, padding=2))


class SeparateResidualFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = in_channels
        self.truc = nn.Hardtanh(-3, 3)
        # Native repeat(axis=0) groups whole filters, rather than interleaving
        # one copy of each filter for each input channel.
        self.kernel = nn.Parameter(
            torch.tensor(
                np.repeat(residual_kernels(), in_channels, axis=0), dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.out_conv[0].weight, a=1)

    def forward(self, features):
        residual = F.conv2d(features, self.kernel, padding=2, groups=self.inc)
        return self.out_conv(self.truc(residual))


class GaussianSmoothing(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = 1
        grids = torch.meshgrid(
            [torch.arange(7, dtype=torch.float32)] * 2, indexing="ij"
        )
        for grid in grids:
            kernel *= (
                1
                / (0.8 * math.sqrt(2 * math.pi))
                * torch.exp(-(((grid - 3) / 0.8) ** 2) / 2)
            )
        kernel = kernel / torch.sum(kernel)
        self.register_buffer("weight", kernel.view(1, 1, 7, 7).repeat(3, 1, 1, 1))

    def forward(self, images):
        return (
            F.conv2d(images, self.weight, groups=3, padding=3)
            if self.training
            else images
        )


class GaussianNoise(nn.Module):
    def forward(self, images):
        if self.training:
            return torch.clamp(images + torch.empty_like(images).normal_(0, 0.1), -1, 1)
        return images
