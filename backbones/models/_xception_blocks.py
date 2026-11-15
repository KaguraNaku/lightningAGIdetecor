"""Shared native Xception building blocks; used by UCF without an IID detector."""

from torch import nn

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
