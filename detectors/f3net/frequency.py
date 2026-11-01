"""Native orthonormal DCT and four overlapping residual frequency filters."""

import numpy as np
import torch
from torch import nn


def dct_matrix(size):
    rows = []
    for frequency in range(size):
        scale = np.sqrt((1.0 if frequency == 0 else 2.0) / size)
        rows.append(
            [
                scale * np.cos((position + 0.5) * np.pi * frequency / size)
                for position in range(size)
            ]
        )
    return torch.tensor(rows).float()


def band_mask(size, start, end):
    return torch.tensor(
        [
            [
                0.0 if row + column > end or row + column < start else 1.0
                for column in range(size)
            ]
            for row in range(size)
        ]
    )


class FrequencyFilter(nn.Module):
    def __init__(self, size, start, end):
        super().__init__()
        self.base = nn.Parameter(band_mask(size, start, end), requires_grad=False)
        self.learnable = nn.Parameter(torch.randn(size, size))
        # Retain both native normal draws; only the second initializes the gain.
        nn.init.normal_(self.learnable, 0.0, 0.1)

    def forward(self, spectrum):
        gain = self.base + (2.0 * torch.sigmoid(self.learnable) - 1.0)
        return spectrum * gain


class FrequencyDecomposition(nn.Module):
    def __init__(self, size):
        super().__init__()
        matrix = dct_matrix(size)
        self._DCT_all = nn.Parameter(matrix, requires_grad=False)
        self._DCT_all_T = nn.Parameter(
            matrix.clone().transpose(0, 1), requires_grad=False
        )
        self.filters = nn.ModuleList(
            [
                FrequencyFilter(size, start, end)
                for start, end in (
                    (0, size // 2.82),
                    (size // 2.82, size // 2),
                    (size // 2, size * 2),
                    (0, size * 2),
                )
            ]
        )

    def forward(self, images):
        spectrum = self._DCT_all @ images @ self._DCT_all_T
        bands = [
            self._DCT_all_T @ frequency_filter(spectrum) @ self._DCT_all
            for frequency_filter in self.filters
        ]
        return torch.cat(bands, dim=1)
