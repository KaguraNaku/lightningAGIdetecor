"""Xception feature/classifier layout compatible with DeepfakeBench states.

Rewritten entry/middle/exit flow. Parameter names intentionally retain the
reference layout; the unused adjustment branch remains for strict state import.
"""

from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, inputs):
        return self.pointwise(self.conv1(inputs))


class Block(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        reps,
        strides=1,
        start_with_relu=True,
        grow_first=True,
    ):
        super().__init__()
        if in_filters != out_filters or strides != 1:
            self.skip = nn.Conv2d(
                in_filters, out_filters, 1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        layers = []
        channels = in_filters
        if grow_first:
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    SeparableConv2d(channels, out_filters, 3, padding=1),
                    nn.BatchNorm2d(out_filters),
                ]
            )
            channels = out_filters
        for _ in range(reps - 1):
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    SeparableConv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                ]
            )
        if not grow_first:
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    SeparableConv2d(in_filters, out_filters, 3, padding=1),
                    nn.BatchNorm2d(out_filters),
                ]
            )
        if start_with_relu:
            layers[0] = nn.ReLU(inplace=False)
        else:
            layers = layers[1:]
        if strides != 1:
            layers.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*layers)

    def forward(self, inputs):
        residual = inputs if self.skip is None else self.skipbn(self.skip(inputs))
        return self.rep(inputs) + residual


class Xception(nn.Module):
    def __init__(self, *, num_classes=2, in_channels=3, mode="original", dropout=0.0):
        super().__init__()
        if mode not in ("original", "shallow_xception"):
            raise ValueError(
                "This Xception baseline supports original/shallow_xception; other reference variants need their own verified specification"
            )
        if isinstance(dropout, bool) and dropout or not 0 <= dropout < 1:
            raise ValueError("Xception dropout must be in [0, 1)")
        self.mode = mode
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        for index in range(4, 12):
            setattr(self, f"block{index}", Block(728, 728, 3))
        self.block12 = Block(728, 1024, 2, 2, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(2048, num_classes)
            )
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=False)
        )

    def features(self, images):
        outputs = self.relu(self.bn1(self.conv1(images)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        for index in range(1, 13):
            if self.mode == "shallow_xception" and 5 <= index <= 11:
                continue
            outputs = getattr(self, f"block{index}")(outputs)
        outputs = self.relu(self.bn3(self.conv3(outputs)))
        return self.bn4(self.conv4(outputs))

    def classifier(self, features):
        features = self.relu(features)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.last_linear(pooled)

    def forward(self, images):
        return self.classifier(self.features(images))


def load_xception(model_id, **kwargs):
    return Xception(**kwargs)
