"""Native B4 stages with three information-attention sites and pooled routes."""

from torch import nn

from .attention import InformationAttention


def projection(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 1), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)
    )


class SIADetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for index, channels in enumerate((24, 32, 56, 112, 160, 272, 448)):
            setattr(
                self,
                f"att{index}conv",
                InformationAttention(channels, spatial=index < 5),
            )
        self.avgpool1 = nn.AdaptiveMaxPool2d((32, 32))
        self.avgpool2 = nn.AdaptiveMaxPool2d((16, 16))
        self.conv1 = projection(32, 56)
        self.conv2 = projection(32, 160)
        self.conv3 = projection(56, 160)
        # The source constructs this unused head after the three residual routes.
        self.linear = nn.Linear(1792, 1)

    def run_blocks(self, features, start, end):
        for index in range(start, end):
            # Native SIA never supplies drop_connect_rate, even while training.
            features = self.backbone.efficientnet._blocks[index](features)
        return features

    def forward_outputs(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("SIA fixed pooled routes require RGB images at 256x256")
        trunk = self.backbone.efficientnet
        features = self.run_blocks(trunk.stem(images), 0, 6)
        features, first = self.att1conv(features)
        residual12 = self.avgpool1(self.conv1(first))
        residual14 = self.avgpool2(self.conv2(first))
        features = self.run_blocks(features, 6, 10)
        features, second = self.att2conv(features + residual12)
        residual24 = self.avgpool2(self.conv3(second))
        features = self.run_blocks(features, 10, 22)
        features, fourth = self.att4conv(features + residual24 + residual14)
        features = trunk.head(self.run_blocks(features, 22, 32))
        logits = self.backbone.classifier(features)
        return {
            "cls": logits,
            "feat": features,
            "attention1": first,
            "attention2": second,
            "attention4": fourth,
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
