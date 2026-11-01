"""Detached native intra/inter distances and batch-global information gains."""

import torch
from torch import nn
from torch.nn import functional as F


class InformationAttention(nn.Module):
    def __init__(self, channels, *, spatial=True, channel=True):
        super().__init__()
        self.outdim = channels
        self.isspace, self.ischannel = spatial, channel
        self.weight = nn.Parameter(torch.ones(channels, 1, 3, 3), requires_grad=False)
        self.weight2 = nn.Parameter(
            torch.ones(channels, 1, 1, 1) * 9, requires_grad=False
        )
        self.band_width = 1.0
        self.channel_range = 5

    def forward(self, features):
        with torch.no_grad():
            batch, channels, height, width = features.shape
            neighbors = F.conv2d(features, self.weight, padding=1, groups=self.outdim)
            centers = F.conv2d(features, self.weight2, groups=self.outdim)
            intra = torch.abs(centers - neighbors)
            wrapped = torch.cat(
                [features, features[:, : self.channel_range + 1]], dim=1
            )
            offsets = [
                (features - wrapped[:, offset : channels + offset]).clone()
                for offset in range(1, self.channel_range + 1)
            ]
            # Source concatenates entire channel-offset tensors, then places the
            # range axis inside channels. This is intentionally not stack/sum.
            distances = (
                torch.cat(offsets, dim=1)
                .view(batch, -1, self.channel_range, height, width)
                .sum(dim=2)
            )
            attention = intra + 0.5 * torch.abs(
                distances.view(batch, -1, height, width)
            )
        if self.ischannel:
            distance = torch.exp(-attention / attention.mean() / 2 / self.band_width**2)
            distance = -torch.log(distance + 0.1)
            gain = (
                distance.view(batch, self.outdim, -1).mean(dim=2).view(batch, -1, 1, 1)
                + 1
            )
        if self.isspace:
            # No epsilon: degenerate zero distances preserve native NaNs.
            space = attention / attention.mean() / 2 / self.band_width**2
            attended = (torch.sigmoid(space) + 1) * features
            if self.ischannel:
                return attended * gain.expand_as(features), space
            return attended, features
        if self.ischannel:
            return features * gain.expand_as(features), features
        raise ValueError("SIA requires spatial or channel attention")
