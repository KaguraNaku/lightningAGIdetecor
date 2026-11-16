"""Inference composition; calibration policies live outside the detector."""

from torch import nn


class LightningDetector(nn.Module):
    def __init__(self, backbone, mapping, classifier):
        super().__init__()
        self.backbone = backbone
        self.mapping = mapping
        self.classifier = classifier

    def forward(self, images):
        features = self.backbone(images)
        projected = self.mapping(features)
        return self.classifier(projected)
