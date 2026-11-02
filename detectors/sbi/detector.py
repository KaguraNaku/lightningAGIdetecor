"""Configured SBI B4 classifier with all three logits and class1 fake score."""

from detectors._shared.detector import BackboneClassifier


class SBIDetector(BackboneClassifier):
    def forward_outputs(self, images):
        features = self.backbone.features(images)
        logits = self.backbone.classifier(features)
        return {"cls": logits, "prob": logits.softmax(1)[:, 1], "feat": features}
