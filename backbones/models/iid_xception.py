"""IID's512-channel Xception variant; baseline Xception remains independent."""

from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, inputs, outputs, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inputs, inputs, kernel_size, stride, padding, groups=inputs, bias=False
        )
        self.pointwise = nn.Conv2d(inputs, outputs, 1, bias=False)

    def forward(self, images):
        return self.pointwise(self.conv1(images))


class Block(nn.Module):
    def __init__(
        self,
        inputs,
        outputs,
        repeats,
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
        self.relu = nn.ReLU(inplace=True)
        layers, channels = [], inputs
        if grow_first:
            layers += [
                self.relu,
                SeparableConv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
            ]
            channels = outputs
        for _ in range(repeats - 1):
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
        skip = images if self.skip is None else self.skipbn(self.skip(images))
        value += skip
        return value


class IIDXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes, self.mode = 2, "adjust_channel"
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
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
        # Native IID constructs this512-input classifier BEFORE the adapter.
        self.last_linear = nn.Linear(512, 2)
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=False)
        )

    def features(self, images):
        value = self.relu(self.bn1(self.conv1(images)))
        value = self.relu(self.bn2(self.conv2(value)))
        for index in range(1, 13):
            value = getattr(self, f"block{index}")(value)
        value = self.relu(self.bn3(self.conv3(value)))
        return self.adjust_channel(self.bn4(self.conv4(value)))

    def classifier(self, features, identity_features=None):
        value = (
            F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            if features.ndim == 4
            else features
        )
        self.last_emb = value
        return self.last_linear(
            value if identity_features is None else value - identity_features
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def load_iid_xception(model_id):
    return IIDXception()
