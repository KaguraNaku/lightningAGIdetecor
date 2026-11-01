"""Independent local timm Xception encoder layout and fan-out initialization.

Kernel primitives are shared with the verified Xception backbone. This variant
retains a 1000-class fc, four activation nodes and the native initialization pass;
it never imports timm at runtime.
"""

from torch import nn
from torch.nn import functional as F

from .xception import Block, SeparableConv2d


class XceptionEncoder(nn.Module):
    def __init__(self, *, num_classes=1000, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        for index in range(4, 12):
            setattr(self, f"block{index}", Block(728, 728, 3))
        self.block12 = Block(728, 1024, 2, 2, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 2048, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.act4 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def features(self, images):
        features = self.act1(self.bn1(self.conv1(images)))
        features = self.act2(self.bn2(self.conv2(features)))
        for index in range(1, 13):
            features = getattr(self, f"block{index}")(features)
        features = self.act3(self.bn3(self.conv3(features)))
        return self.act4(self.bn4(self.conv4(features)))

    def forward(self, images):
        return self.fc(F.adaptive_avg_pool2d(self.features(images), 1).flatten(1))


def load_xception_encoder(model_id, **kwargs):
    return XceptionEncoder(**kwargs)
