"""UCF's native adjusted512 features with its retained unused2048-input head."""

from torch import nn

from ._xception_blocks import Block, SeparableConv2d


class UCFXception(nn.Module):
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
        # Distinct from IID's512-input classifier; retain native RNG and state.
        self.last_linear = nn.Linear(2048, 2)
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

    def forward(self, images):
        return self.features(images)


def load_ucf_xception(model_id):
    return UCFXception()
