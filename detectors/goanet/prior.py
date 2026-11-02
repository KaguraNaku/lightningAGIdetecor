"""Native double-normal CDC and fixed-sign high-band prior extraction."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class CDCConv(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        self.theta = 0.7

    def forward(self, images):
        normal = self.conv(images)
        if math.fabs(self.theta) < 1e-8:
            return normal
        difference = (
            self.conv.weight.sum(2, keepdim=True)
            .sum(3, keepdim=True)
            .expand_as(self.conv.weight)
        )
        extra = F.conv2d(
            images,
            self.conv.weight - difference * self.theta,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )
        return normal + extra


class DWTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        filters = {
            "ll_filter": [[0.5, 0.5], [0.5, 0.5]],
            "lh_filter": [[-0.5, -0.5], [0.5, 0.5]],
            "hl_filter": [[-0.5, 0.5], [-0.5, 0.5]],
            "hh_filter": [[-0.5, 0.5], [0.5, -0.5]],
        }
        for name, values in filters.items():
            self.register_buffer(
                name, torch.tensor(values).view(1, 1, 2, 2).repeat(3, 1, 1, 1)
            )

    def forward(self, images):
        # The unused LL buffer remains in the native checkpoint layout.
        F.conv2d(images, self.ll_filter, stride=2, groups=images.size(1))
        bands = [
            F.conv2d(images, getattr(self, name), stride=2, groups=images.size(1))
            for name in ("lh_filter", "hl_filter", "hh_filter")
        ]
        return torch.cat(bands, dim=1)


class PriorExtractionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cdc1 = CDCConv(3, 32)
        self.cdc2 = CDCConv(32, 64)
        self.dwt = DWTExtractor()
        self.dwt_conv = nn.Conv2d(9, 64, 3, padding=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images):
        cdc = F.relu(self.cdc1(images))
        cdc = F.relu(self.cdc2(cdc))
        bands = F.relu(self.dwt_conv(self.dwt(images)))
        bands = F.interpolate(
            bands, size=cdc.shape[2:], mode="bilinear", align_corners=False
        )
        return self.fusion(torch.cat([cdc, bands], dim=1)).squeeze(-1).squeeze(-1)
