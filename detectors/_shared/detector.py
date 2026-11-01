"""A native feature/classifier backbone composed into a logit-only detector."""

from torch import nn


class BackboneClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        return self.backbone.classifier(self.backbone.features(images))
