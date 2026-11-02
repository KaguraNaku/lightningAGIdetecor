"""IID's trainable IResNet50, including its batch-size-conditional final BN."""

from contextlib import nullcontext

import torch
from torch import nn


class IBasicBlock(nn.Module):
    def __init__(self, inputs, outputs, stride, downsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inputs, eps=1e-5)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs, eps=1e-5)
        self.prelu = nn.PReLU(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outputs, eps=1e-5)
        self.downsample, self.stride = downsample, stride

    def forward(self, images):
        value = self.conv1(self.bn1(images))
        value = self.conv2(self.prelu(self.bn2(value)))
        value = self.bn3(value)
        value += images if self.downsample is None else self.downsample(images)
        return value


class IResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16, self.inplanes = False, 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        for index, (channels, repeats) in enumerate(
            zip((64, 128, 256, 512), (3, 4, 14, 3), strict=True), 1
        ):
            setattr(self, f"layer{index}", self._layer(channels, repeats))
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(0, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0, 0.1)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def _layer(self, channels, repeats):
        # The native constructor initializes the downsample before block convs.
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, channels, 1, 2, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5),
        )
        layers = [IBasicBlock(self.inplanes, channels, 2, downsample)]
        self.inplanes = channels
        layers += [IBasicBlock(channels, channels, 1, None) for _ in range(repeats - 1)]
        return nn.Sequential(*layers)

    def forward(self, images):
        # Native fp16=False disables the enclosing CUDA autocast for this trunk;
        # fc/final BN are outside that context. CPU transport has no CUDA effect.
        context = (
            torch.autocast("cuda", enabled=False)
            if images.device.type == "cuda"
            else nullcontext()
        )
        with context:
            value = self.prelu(self.bn1(self.conv1(images)))
            for index in range(1, 5):
                value = getattr(self, f"layer{index}")(value)
            value = self.dropout(torch.flatten(self.bn2(value), 1))
        value = self.fc(value)
        return self.features(value) if value.size(0) > 1 else value


def load_iresnet50_iid(model_id):
    return IResNet50()
