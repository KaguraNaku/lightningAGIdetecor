"""Directional CDC, grouped RMS scaling and native global wavelet mapping."""

import torch
from torch import nn
from torch.nn import functional as F

from .wavelets import HaarPyramid


class DilatedCDC(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.padding = dilation
        self.weight = nn.Parameter(torch.empty(3, 3, 3, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(9))
        self.group_norm = nn.GroupNorm(3, 9)
        self.offsets = [
            ((i - 1) * dilation, (j - 1) * dilation)
            for i in range(3)
            for j in range(3)
            if (i, j) != (1, 1)
        ]

    def forward(self, values):
        batch, _, directions, height, width = values.shape
        if directions != 3:
            raise ValueError("Effluxlet Haar CDC requires three directions")
        pad = self.padding
        center = F.pad(values, [pad] * 6, mode="replicate")
        outputs = torch.zeros(batch, 9, height, width, device=values.device)
        for direction in range(3):
            original = values[:, :, direction]
            center_direction = center[:, :, direction + pad]
            result = torch.zeros(batch, 3, height, width, device=values.device)
            for dx, dy in self.offsets:
                shifted = F.pad(
                    original, [pad - dy, pad + dy, pad - dx, pad + dx], mode="replicate"
                )
                difference = (shifted - center_direction)[..., pad:-pad, pad:-pad]
                weights = (
                    self.weight[direction, :, :, 1 + dx // pad, 1 + dy // pad]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                result += F.conv2d(difference, weights)
            outputs[:, direction * 3 : (direction + 1) * 3] = result
        return self.group_norm(outputs + self.bias.view(1, -1, 1, 1))


class GroupedL2Pool2d(nn.Module):
    def forward(self, values):
        batch, channels, directions, height, width = values.shape
        grouped = (
            values.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(batch, directions, channels, height, width)
        )
        grouped = grouped * torch.sqrt(
            F.adaptive_avg_pool3d(grouped**2, (1, 1, 1)) + 1e-8
        )
        return (
            grouped.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(batch, channels, directions, height, width)
        )


class FeatureMapping(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((3, 1, 1))
        self.projection = nn.Sequential(
            nn.Linear(9, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, dimension),
        )

    def forward(self, values):
        return self.projection(
            self.global_pool(values).view(
                values.shape[0], values.shape[1] * values.shape[2]
            )
        )


class WaveletsEnhancement(nn.Module):
    def __init__(self, dimension=1024):
        super().__init__()
        self.transform = HaarPyramid()
        self.cdc_conv = nn.ModuleList([DilatedCDC(i) for i in range(1, 5)])
        self.grouped_pooled = GroupedL2Pool2d()
        self.feature_mapping = nn.ModuleList(
            [FeatureMapping(dimension) for _ in range(4)]
        )

    def forward(self, images):
        _, highs = self.transform(images)
        outputs = []
        for index, high in enumerate(highs):
            batch, channels, directions, height, width = high.shape
            residual = (
                self.cdc_conv[index](high)
                .unsqueeze(2)
                .expand(-1, -1, directions, -1, -1)
            )
            count = residual.shape[1]
            if count != channels:
                residual = residual.permute(0, 2, 3, 4, 1).contiguous()
                residual = F.adaptive_avg_pool1d(
                    residual.view(-1, count).unsqueeze(1), channels
                ).squeeze(1)
                residual = (
                    residual.view(batch, directions, height, width, channels)
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                )
            outputs.append(
                self.feature_mapping[index](high + self.grouped_pooled(residual))
            )
        return outputs
