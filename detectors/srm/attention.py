"""Native spatial, cross-modal and channel fusion operations and RNG order."""

import torch
from torch import nn

from .filters import ResidualFilter


def xavier_weights(module):
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(child.weight, gain=0.02)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        xavier_weights(self)

    def forward(self, features):
        return self.sigmoid(
            self.sharedMLP(self.avg_pool(features))
            + self.sharedMLP(self.max_pool(features))
        )


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        xavier_weights(self)

    def forward(self, features):
        mean = torch.mean(features, dim=1, keepdim=True)
        maximum, _ = torch.max(features, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([mean, maximum], dim=1)))


class PixelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = ResidualFilter()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pa = SpatialAttention()
        # Deliberately resets the nested spatial convolution after its Xavier draw.
        for child in self.modules():
            if isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight, a=1)
                if child.bias is not None:
                    nn.init.constant_(child.bias, 0)

    def forward(self, residual):
        return self.pa(self.conv(self.srm(residual)))


class DualCrossModalAttention(nn.Module):
    def __init__(self, channels=728, size=16):
        super().__init__()
        self.key_conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv2 = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv_share = nn.Conv2d(channels // 8, channels // 8, 1)
        self.linear1 = nn.Linear(size * size, size * size)
        self.linear2 = nn.Linear(size * size, size * size)
        self.value_conv1 = nn.Conv2d(channels, channels, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.value_conv2 = nn.Conv2d(channels, channels, 1)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        xavier_weights(self)

    def forward(self, x, y):
        batch, channels, height, width = x.shape
        positions = height * width
        key1 = (
            self.key_conv_share(self.key_conv1(x))
            .view(batch, -1, positions)
            .permute(0, 2, 1)
        )
        key2 = self.key_conv_share(self.key_conv2(y)).view(batch, -1, positions)
        energy = torch.bmm(key1, key2)
        attention1 = self.softmax(self.linear1(energy))
        attention2 = self.softmax(self.linear2(energy.permute(0, 2, 1)))
        value_y = self.value_conv2(y).view(batch, -1, positions)
        crossed_y = torch.bmm(value_y, attention1.permute(0, 2, 1)).view(
            batch, channels, height, width
        )
        out_x = self.gamma1 * crossed_y + x
        value_x = self.value_conv1(x).view(batch, -1, positions)
        crossed_x = torch.bmm(value_x, attention2.permute(0, 2, 1)).view(
            batch, channels, height, width
        )
        return out_x, self.gamma2 * crossed_x + y


class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(4096, 2048, 1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
        self.ca = ChannelAttention(2048)
        # Native init_weight inspects direct children, neither of which is a
        # Conv2d; preserve those default convblk weights rather than resetting.

    def forward(self, x, y):
        fused = self.convblk(torch.cat((x, y), dim=1))
        return fused * self.ca(fused)
