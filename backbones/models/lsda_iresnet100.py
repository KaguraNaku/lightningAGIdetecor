"""Native LSDA content encoder: partial freezing, train-mode BN, spatial output."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .iresnet50_iid import IBasicBlock


class LSDAIResNet100(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16, self.inplanes = False, 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._layer(64, 3)
        self.layer2 = self._layer(128, 13)
        self.layer3 = self._layer(256, 30)
        for layer in (self.layer1, self.layer2, self.layer3):
            for parameter in layer.parameters():
                parameter.requires_grad = False
        self.layer4 = self._layer(512, 3)
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
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, channels, 1, 2, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5),
        )
        layers = [IBasicBlock(self.inplanes, channels, 2, downsample)]
        self.inplanes = channels
        layers += [IBasicBlock(channels, channels, 1, None) for _ in range(repeats - 1)]
        return nn.Sequential(*layers)

    def forward(self, images):
        context = (
            torch.autocast("cuda", enabled=False)
            if images.device.type == "cuda"
            else nullcontext()
        )
        with context:
            value = self.prelu(self.bn1(self.conv1(images)))
            for index in range(1, 5):
                value = getattr(self, f"layer{index}")(value)
            value = self.dropout(self.bn2(value))
            return F.avg_pool2d(value, kernel_size=2, stride=2, padding=0)


def load_lsda_iresnet100(model_id):
    return LSDAIResNet100()
