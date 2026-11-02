"""Native SLADD Xception, retaining its unused type/magnitude/map heads for PCL."""

from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(
        self, inputs, outputs, kernel_size=1, stride=1, padding=0, *, bias=False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inputs, inputs, kernel_size, stride, padding, groups=inputs, bias=bias
        )
        self.pointwise = nn.Conv2d(inputs, outputs, 1, bias=bias)

    def forward(self, images):
        return self.pointwise(self.conv1(images))


class Block(nn.Module):
    def __init__(
        self,
        inputs,
        outputs,
        repetitions,
        stride=1,
        *,
        start_with_relu=True,
        grow_first=True,
    ):
        super().__init__()
        if inputs != outputs or stride != 1:
            self.skip = nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(outputs)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=False)
        layers, channels = [], inputs
        if grow_first:
            layers += [
                self.relu,
                SeparableConv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
            ]
            channels = outputs
        for _ in range(repetitions - 1):
            layers += [
                self.relu,
                SeparableConv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            ]
        if not grow_first:
            layers += [
                self.relu,
                SeparableConv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
            ]
        if start_with_relu:
            layers[0] = nn.ReLU(inplace=False)
        else:
            layers = layers[1:]
        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*layers)

    def forward(self, images):
        value = self.rep(images)
        residual = images if self.skip is None else self.skipbn(self.skip(images))
        value += residual
        return value


class RegressionMap(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = SeparableConv2d(channels, 1, 3, padding=1)
        self.s = nn.Sigmoid()

    def forward(self, value):
        return self.s(self.c(value))


class XceptionSLADD(nn.Module):
    def __init__(self, *, num_classes=2, in_channels=3, dropout=0.0):
        super().__init__()
        if isinstance(dropout, bool) and dropout or not 0 <= dropout < 1:
            raise ValueError("Xception SLADD dropout must be in[0,1)")
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)
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
        self.type_fc = nn.Linear(2048, 5)
        self.mag_fc = nn.Linear(2048, 1)
        self.map = RegressionMap(728)
        self.pecent = 1.0 / 1.5

    def features(self, images):
        value = self.relu(self.bn1(self.conv1(images)))
        value = self.relu(self.bn2(self.conv2(value)))
        for index in range(1, 8):
            value = getattr(self, f"block{index}")(value)
        spatial = value
        for index in range(8, 13):
            value = getattr(self, f"block{index}")(value)
        value = self.relu(self.bn3(self.conv3(value)))
        return self.bn4(self.conv4(value)), spatial

    def classifier(self, features):
        pooled = F.adaptive_avg_pool2d(self.relu(features), (1, 1)).view(
            features.size(0), -1
        )
        return self.last_linear(pooled), pooled

    def forward(self, images):
        # PCL consumes features/classifier explicitly and does not activate the
        # extra native heads. Keep the backbone's ordinary classifier interface.
        return self.classifier(self.features(images)[0])[0]


def load_xception_sladd(model_id, **kwargs):
    return XceptionSLADD(**kwargs)
