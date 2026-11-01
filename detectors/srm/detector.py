"""Two Xception branches with residual guidance and fixed-grid cross attention."""

from torch import nn

from .attention import DualCrossModalAttention, FeatureFusion, PixelAttention
from .filters import (
    GaussianNoise,
    GaussianSmoothing,
    ResidualFilter,
    SeparateResidualFilter,
)


def blocks(backbone, features, first, last):
    for index in range(first, last + 1):
        if backbone.mode == "shallow_xception" and 4 <= index <= 11:
            continue
        features = getattr(backbone, f"block{index}")(features)
    return features


def exit_features(backbone, features):
    features = backbone.relu(backbone.bn3(backbone.conv3(features)))
    return backbone.bn4(backbone.conv4(features))


class SRMDetector(nn.Module):
    def __init__(self, rgb, residual):
        super().__init__()
        self.backbone_rgb, self.backbone_srm = rgb, residual
        # Kept for checkpoint compatibility. Native features never invokes them.
        self.noise, self.blur = GaussianNoise(), GaussianSmoothing()
        self.srm_conv0 = ResidualFilter()
        self.srm_conv1 = SeparateResidualFilter(32, 32)
        self.srm_conv2 = SeparateResidualFilter(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.att_map = None
        self.srm_sa = PixelAttention()
        self.srm_sa_post = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dual_cma0 = DualCrossModalAttention()
        self.dual_cma1 = DualCrossModalAttention()
        self.fusion = FeatureFusion()

    def features(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("SRM fixed 16x16 attention requires RGB images at 256x256")
        rgb, residual = self.backbone_rgb, self.backbone_srm
        srm = self.srm_conv0(images)
        x = rgb.relu(rgb.bn1(rgb.conv1(images)))
        y = self.relu(
            residual.relu(residual.bn1(residual.conv1(srm))) + self.srm_conv1(x)
        )
        x = rgb.relu(rgb.bn2(rgb.conv2(x)))
        y = self.relu(
            residual.relu(residual.bn2(residual.conv2(y))) + self.srm_conv2(x)
        )
        self.att_map = self.srm_sa(srm)
        x = self.srm_sa_post(x * self.att_map + x)
        x, y = self.dual_cma0(blocks(rgb, x, 1, 3), blocks(residual, y, 1, 3))
        x, y = self.dual_cma1(blocks(rgb, x, 4, 7), blocks(residual, y, 4, 7))
        x, y = blocks(rgb, x, 8, 12), blocks(residual, y, 8, 12)
        return self.fusion(exit_features(rgb, x), exit_features(residual, y))

    def forward_outputs(self, images):
        features = self.features(images)
        logits = self.backbone_rgb.classifier(features)
        return {"cls": logits, "feat": features, "attention": self.att_map}

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
