"""Full native six-channel SLADD policy encoder and ImageNet initialization.

This encoder deliberately has different depthwise names and in-place ReLUs
from the detector's Xception backbone. Its partial ImageNet import and second
stem construction preserve the source initialization, including fresh weights.
"""

from torch import nn
from torch.nn import functional as F

from detectors._shared.checkpoint import read_weights


class SeparableConv2d(nn.Module):
    def __init__(self, inputs, outputs, kernel_size, stride=1, padding=0):
        super().__init__()
        self.c = nn.Conv2d(
            inputs, inputs, kernel_size, stride, padding, groups=inputs, bias=False
        )
        self.pointwise = nn.Conv2d(inputs, outputs, 1, bias=False)

    def forward(self, images):
        return self.pointwise(self.c(images))


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
        self.relu = nn.ReLU(inplace=True)
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


class SynthesisEncoder(nn.Module):
    def __init__(self, *, num_region=10, num_type=4, num_mag=1, in_channels=6):
        super().__init__()
        self.num_region, self.num_type, self.num_mag = num_region, num_type, num_mag
        self.iniconv = nn.Conv2d(in_channels, 32, 3, 2, bias=False)
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
        self.fc_region = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_region))
        self.fc_type = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_type))
        self.fc_mag = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, num_mag))

    def features(self, images):
        value = self.relu(self.bn1(self.iniconv(images)))
        value = self.relu(self.bn2(self.conv2(value)))
        for index in range(1, 13):
            value = getattr(self, f"block{index}")(value)
        value = self.relu(self.bn3(self.conv3(value)))
        return self.bn4(self.conv4(value))

    def forward(self, images):
        value = self.relu(self.features(images))
        pooled = F.adaptive_avg_pool2d(value, (1, 1)).view(value.size(0), -1)
        return self.fc_region(pooled), self.fc_type(pooled), self.fc_mag(pooled)


class SynthesisTransfer(nn.Module):
    """Selected source TransferModel profile, with explicit pretrained path.

    It always loads ImageNet initialization before replacing the six-channel
    stem. The receipt reports every unmatched name; depthwise '.conv1' weights
    are intentionally not renamed to '.c'. This is not a detector checkpoint.
    """

    def __init__(
        self, pretrained, *, num_region=10, num_type=4, num_mag=1, in_channels=6
    ):
        super().__init__()
        self.model = SynthesisEncoder(
            num_region=num_region,
            num_type=num_type,
            num_mag=num_mag,
            in_channels=in_channels,
        )
        weights, receipt = read_weights(pretrained)
        adapted = {
            name: value.unsqueeze(-1).unsqueeze(-1) if "pointwise" in name else value
            for name, value in weights.items()
        }
        result = self.model.load_state_dict(adapted, strict=False)
        self.initialization = {
            **receipt,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "matched_supplied_keys": len(set(adapted) & set(self.model.state_dict())),
            "depthwise_names_translated": False,
            "second_stem_construction": in_channels != 3,
        }
        if in_channels != 3:
            self.model.iniconv = nn.Conv2d(in_channels, 32, 3, 2, bias=False)
            nn.init.xavier_normal_(self.model.iniconv.weight.data, gain=0.02)

    def features(self, images):
        return self.model.features(images)

    def forward(self, images):
        return self.model(images)
