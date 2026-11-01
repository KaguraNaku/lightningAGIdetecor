"""Xception logits and the separate pooled positive CORE representation."""

from torch import nn
from torch.nn import functional as F


class COREDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward_outputs(self, images):
        features = self.backbone.features(images)
        core = F.adaptive_avg_pool2d(F.relu(features, inplace=False), 1).flatten(1)
        logits = self.backbone.classifier(features)
        return {"cls": logits, "feat": features, "core_feat": core}

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
