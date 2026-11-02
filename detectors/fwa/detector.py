"""Native original-Xception FWA classifier; generation belongs to collation."""

from detectors._shared.detector import BackboneClassifier


class FWADetector(BackboneClassifier):
    def forward_outputs(self, images):
        features = self.backbone.features(images)
        logits = self.backbone.classifier(features)
        return {"cls": logits, "prob": logits.softmax(1)[:, 1], "feat": features}
