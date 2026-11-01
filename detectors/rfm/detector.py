"""RFM inference exposes logits; augmentation and its backward live in Trainer."""

from torch import nn


class RFMDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward_outputs(self, images):
        features = self.backbone.features(images)
        return {"cls": self.backbone.classifier(features), "feat": features}

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
