"""Independent B4 kernels preserving the local wrapper's static-padding rules."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return inputs * torch.sigmoid(inputs)

    @staticmethod
    def backward(ctx, gradient):
        (inputs,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(inputs)
        return gradient * (sigmoid * (1 + inputs * (1 - sigmoid)))


class Swish(nn.Module):
    def forward(self, inputs):
        return _SwishFunction.apply(inputs)


class StaticConv2d(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, *, image_size, stride=1, **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, **kwargs
        )
        padding = max(
            (math.ceil(image_size / stride) - 1) * stride + kernel_size - image_size, 0
        )
        half = padding // 2
        self.static_padding = (
            nn.ZeroPad2d((half, padding - half, half, padding - half))
            if padding
            else nn.Identity()
        )

    def forward(self, inputs):
        return F.conv2d(
            self.static_padding(inputs),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def normalization(channels):
    return nn.BatchNorm2d(channels, momentum=1 - 0.99, eps=1e-3)


class MBConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel, expansion, stride, image_size
    ):
        super().__init__()
        hidden = in_channels * expansion
        self.expansion = expansion
        self.skip = stride == 1 and in_channels == out_channels
        if expansion != 1:
            self._expand_conv = StaticConv2d(
                in_channels, hidden, 1, image_size=image_size, bias=False
            )
            self._bn0 = normalization(hidden)
        self._depthwise_conv = StaticConv2d(
            hidden,
            hidden,
            kernel,
            image_size=image_size,
            stride=stride,
            groups=hidden,
            bias=False,
        )
        self._bn1 = normalization(hidden)
        output_size = math.ceil(image_size / stride)
        squeezed = max(1, int(in_channels * 0.25))
        self._se_reduce = StaticConv2d(hidden, squeezed, 1, image_size=1)
        self._se_expand = StaticConv2d(squeezed, hidden, 1, image_size=1)
        self._project_conv = StaticConv2d(
            hidden, out_channels, 1, image_size=output_size, bias=False
        )
        self._bn2 = normalization(out_channels)
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=0.0):
        outputs = inputs
        if self.expansion != 1:
            outputs = self._swish(self._bn0(self._expand_conv(outputs)))
        outputs = self._swish(self._bn1(self._depthwise_conv(outputs)))
        squeezed = self._swish(self._se_reduce(F.adaptive_avg_pool2d(outputs, 1)))
        outputs = torch.sigmoid(self._se_expand(squeezed)) * outputs
        outputs = self._bn2(self._project_conv(outputs))
        if self.skip:
            if self.training and drop_connect_rate:
                keep = 1 - drop_connect_rate
                mask = torch.floor(
                    keep
                    + torch.rand(
                        [inputs.shape[0], 1, 1, 1],
                        dtype=inputs.dtype,
                        device=inputs.device,
                    )
                )
                outputs = outputs / keep * mask
            outputs = outputs + inputs
        return outputs


class B4Trunk(nn.Module):
    # Rounded B4 widths and repeats (width=1.4, depth=1.8, divisor=8).
    STAGES = (
        (48, 24, 3, 1, 1, 2),
        (24, 32, 3, 6, 2, 4),
        (32, 56, 5, 6, 2, 4),
        (56, 112, 3, 6, 2, 6),
        (112, 160, 5, 6, 1, 6),
        (160, 272, 5, 6, 2, 8),
        (272, 448, 3, 6, 1, 2),
    )

    def __init__(self):
        super().__init__()
        self._conv_stem = StaticConv2d(3, 48, 3, stride=2, image_size=380, bias=False)
        self._bn0 = normalization(48)
        self._blocks = nn.ModuleList()
        image_size = 190
        for incoming, outgoing, kernel, expansion, stride, repeats in self.STAGES:
            self._blocks.append(
                MBConv(incoming, outgoing, kernel, expansion, stride, image_size)
            )
            image_size = math.ceil(image_size / stride)
            for _ in range(repeats - 1):
                self._blocks.append(
                    MBConv(outgoing, outgoing, kernel, expansion, 1, image_size)
                )
        self._conv_head = StaticConv2d(448, 1792, 1, image_size=image_size, bias=False)
        self._bn1 = normalization(1792)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.4)
        # Native construction draws this unused head before replacing the stem.
        self._fc = nn.Linear(1792, 1000)
        self._swish = Swish()

    def stem(self, images):
        return self._swish(self._bn0(self._conv_stem(images)))

    def blocks(self, features, start=0, end=None):
        for index in range(start, len(self._blocks) if end is None else end):
            features = self._blocks[index](
                features, drop_connect_rate=0.2 * (float(index) / len(self._blocks))
            )
        return features

    def head(self, features):
        return self._swish(self._bn1(self._conv_head(features)))

    def extract_features(self, images):
        return self.head(self.blocks(self.stem(images)))

    def extract_endpoints(self, images):
        outputs = self.stem(images)
        endpoints = {}
        for index in range(len(self._blocks)):
            previous = outputs
            outputs = self.blocks(outputs, index, index + 1)
            if previous.shape[2] > outputs.shape[2]:
                endpoints[f"reduction_{len(endpoints) + 1}"] = previous
            elif index == len(self._blocks) - 1:
                endpoints[f"reduction_{len(endpoints) + 1}"] = outputs
        endpoints[f"reduction_{len(endpoints) + 1}"] = self.head(outputs)
        return endpoints

    def forward(self, images):
        return self._fc(
            self._dropout(self._avg_pooling(self.extract_features(images)).flatten(1))
        )


class EfficientNetB4(nn.Module):
    def __init__(self, *, num_classes=2, in_channels=3, mode="Original", dropout=0.0):
        super().__init__()
        if mode not in ("Original", "original"):
            raise ValueError(
                "B4 baseline supports Original mode; the local adjust_channel classifier has incompatible dimensions"
            )
        if isinstance(dropout, bool) and dropout or not 0 <= dropout <= 1:
            raise ValueError("EfficientNet-B4 dropout must be a probability")
        self.efficientnet = B4Trunk()
        # Deliberate native behavior: discard even a pretrained stem, replace
        # static SAME with a fresh, unpadded convolution, retain the 380 schedule.
        self.efficientnet._conv_stem = nn.Conv2d(
            in_channels, 48, 3, stride=2, bias=False
        )
        self.efficientnet._fc = nn.Identity()
        self.dropout = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        self.last_layer = nn.Linear(1792, num_classes)

    def features(self, images):
        return self.efficientnet.extract_features(images)

    def classifier(self, features):
        outputs = F.adaptive_avg_pool2d(features, 1).flatten(1)
        if self.dropout:
            outputs = self.dropout_layer(outputs)
        self.last_emb = outputs
        return self.last_layer(outputs)

    def block_part1(self, images):
        return self.efficientnet.blocks(self.efficientnet.stem(images), 0, 10)

    def block_part2(self, features):
        return self.efficientnet.blocks(features, 10, 22)

    def block_part3(self, features):
        return self.efficientnet.head(self.efficientnet.blocks(features, 22))

    def end_points(self, images):
        return self.efficientnet.extract_endpoints(images)

    def forward(self, images):
        return self.classifier(self.features(images))


def load_efficientnetb4(model_id, **kwargs):
    return EfficientNetB4(**kwargs)
