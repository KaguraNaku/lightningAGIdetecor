"""Four RGB frequency bands feed the local twelve-channel Xception."""

from torch import nn

from .frequency import FrequencyDecomposition


class F3NetDetector(nn.Module):
    def __init__(self, backbone, image_size):
        super().__init__()
        self.backbone = backbone
        self.FAD_head = FrequencyDecomposition(image_size)

    def forward_outputs(self, images):
        fad = self.FAD_head(images)
        features = self.backbone.features(fad)
        logits = self.backbone.classifier(features)
        return {"cls": logits, "feat": features, "fad": fad}

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
