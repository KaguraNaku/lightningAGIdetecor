"""Native attention maps, bilinear pooling and V1 texture enhancement."""

import torch
from torch import nn
from torch.nn import functional as F


class AttentionMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.zeros(1, 1, 24, 24))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.conv_extract = nn.Conv2d(160, 160, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(160)
        self.conv2 = nn.Conv2d(160, 4, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, features):
        features = F.relu(self.bn1(self.conv_extract(features)), inplace=True)
        features = F.elu(self.bn2(self.conv2(features))) + 1
        return features * F.interpolate(self.mask, features.shape[-2:], mode="nearest")


class AttentionPooling(nn.Module):
    def forward(self, features, attentions, norm=2):
        if features.shape[-2:] != attentions.shape[-2:]:
            attentions = F.interpolate(
                attentions,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        matrix = torch.einsum("imjk,injk->imn", attentions, features)
        if norm == 1:
            matrix /= torch.sum(attentions + 1e-8, dim=(2, 3)).unsqueeze(-1)
        elif norm == 2:
            matrix = F.normalize(matrix, p=2, dim=-1)
        return matrix


class TextureEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(32, 32, 1)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(96, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv_last = nn.Conv2d(128, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn_last = nn.BatchNorm2d(128)

    def forward(self, features, attentions):
        reduced = F.adaptive_avg_pool2d(features, attentions.shape[-2:])
        residual = features - F.interpolate(
            reduced, features.shape[-2:], mode="nearest"
        )
        first = self.conv0(residual)
        second = self.conv1(F.relu(self.bn1(first), inplace=True))
        second = torch.cat([first, second], dim=1)
        third = self.conv2(F.relu(self.bn2(second), inplace=True))
        third = torch.cat([second, third], dim=1)
        fourth = self.conv3(F.relu(self.bn3(third), inplace=True))
        fourth = torch.cat([third, fourth], dim=1)
        enhanced = self.bn_last(self.conv_last(F.relu(self.bn4(fourth), inplace=True)))
        return enhanced, reduced
