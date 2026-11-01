"""Independent Meso networks retaining the local reference's state layout.

BatchNorm instances are intentionally reused at multiple depths. The reference
uses Dropout2d on flattened features; keep that operation, including its current
PyTorch deprecation warning, rather than silently changing the stochastic rule.
"""

import torch
from torch import nn


class _MesoHead(nn.Module):
    def _build_head(self, num_classes):
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def classifier(self, features):
        return self.fc2(self.dropout(self.leakyrelu(self.fc1(self.dropout(features)))))

    def forward(self, images):
        return self.classifier(self.features(images))


class Meso4(_MesoHead):
    def __init__(self, *, num_classes=2, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.maxpooling2 = nn.MaxPool2d(4)
        self._build_head(num_classes)

    def features(self, images):
        outputs = images
        for index in range(1, 5):
            convolution = getattr(self, f"conv{index}")
            normalization = self.bn1 if index <= 2 else self.bn2
            pool = self.maxpooling2 if index == 4 else self.maxpooling1
            outputs = pool(normalization(self.relu(convolution(outputs))))
        return outputs.flatten(1)


class MesoInception4(_MesoHead):
    def __init__(self, *, num_classes=2, in_channels=3):
        super().__init__()
        if in_channels != 3:
            raise ValueError("The reference MesoInception4 branches require RGB input")
        for index, channels, first, fourth in ((1, 3, 1, 2), (2, 11, 2, 2)):
            prefix = f"Incption{index}_"
            # Preserve the historical spelling for strict raw-state import.
            setattr(self, prefix + "conv1", nn.Conv2d(channels, first, 1, bias=False))
            for branch, width in ((2, 4), (3, 4), (4, fourth)):
                dilation = branch - 1
                setattr(
                    self,
                    prefix + f"conv{branch}_1",
                    nn.Conv2d(channels, width, 1, bias=False),
                )
                setattr(
                    self,
                    prefix + f"conv{branch}_2",
                    nn.Conv2d(
                        width, width, 3, padding=dilation, dilation=dilation, bias=False
                    ),
                )
            setattr(self, prefix + "bn", nn.BatchNorm2d(first + 8 + fourth))
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(4)
        self._build_head(num_classes)

    def _inception(self, images, index):
        prefix = f"Incption{index}_"
        branches = [getattr(self, prefix + "conv1")(images)]
        for branch in (2, 3, 4):
            outputs = getattr(self, prefix + f"conv{branch}_1")(images)
            branches.append(getattr(self, prefix + f"conv{branch}_2")(outputs))
        return self.maxpooling1(getattr(self, prefix + "bn")(torch.cat(branches, 1)))

    def features(self, images):
        outputs = self._inception(self._inception(images, 1), 2)
        outputs = self.maxpooling1(self.bn1(self.relu(self.conv1(outputs))))
        outputs = self.maxpooling2(self.bn1(self.relu(self.conv2(outputs))))
        return outputs.flatten(1)


def load_meso4(model_id, **kwargs):
    return Meso4(**kwargs)


def load_meso_inception4(model_id, **kwargs):
    return MesoInception4(**kwargs)
