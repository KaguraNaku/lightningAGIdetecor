"""Independent ResNet34 with the local benchmark's Sequential checkpoint names."""

from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, *, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, images):
        outputs = self.relu(self.bn1(self.conv1(images)))
        outputs = self.bn2(self.conv2(outputs))
        identity = images if self.downsample is None else self.downsample(images)
        outputs += identity
        return self.relu(outputs)


def stage(in_channels, channels, blocks, stride):
    downsample = None
    if stride != 1 or in_channels != channels:
        # Native construction draws these weights before the residual branch.
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False),
            nn.BatchNorm2d(channels),
        )
    layers = [BasicBlock(in_channels, channels, stride=stride, downsample=downsample)]
    layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class ResNet34(nn.Module):
    def __init__(self, *, num_classes=2, in_channels=3, mode="Original"):
        super().__init__()
        if in_channels != 3:
            raise ValueError("The local ResNet34 reference always uses RGB input")
        if mode not in ("Original", "original", "adjust_channel"):
            raise ValueError("Unsupported ResNet34 mode")
        self.mode = mode
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            stage(64, 64, 3, 1),
            stage(64, 128, 4, 2),
            stage(128, 256, 6, 2),
            stage(256, 512, 3, 2),
        )
        # torchvision constructs its unused ImageNet head before resetting the
        # trunk's convolution weights. Retain its RNG draws for seed parity.
        discarded_head = nn.Linear(512, 1000)
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        del discarded_head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        if mode == "adjust_channel":
            # The local reference saves this branch but does not use it in features.
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
            )

    def features(self, images):
        return self.resnet(images)

    def classifier(self, features):
        return self.fc(self.avgpool(features).flatten(1))

    def forward(self, images):
        return self.classifier(self.features(images))


def load_resnet34(model_id, **kwargs):
    return ResNet34(**kwargs)
