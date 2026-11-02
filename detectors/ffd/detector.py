"""Native block7 sigmoid attention, followed by the original Xception exit."""

from torch import nn
from torch.nn import functional as F

from backbones.models.xception import SeparableConv2d


class RegressionMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = SeparableConv2d(728, 1, 3, stride=1, padding=1, bias=False)
        self.s = nn.Sigmoid()

    def forward(self, features):
        return self.s(self.c(features)), None


class FFDDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.map = RegressionMap()

    def forward_outputs(self, images):
        b = self.backbone
        features = b.relu(b.bn1(b.conv1(images)))
        features = b.relu(b.bn2(b.conv2(features)))
        for index in range(1, 8):
            features = getattr(b, f"block{index}")(features)
        mask, vector = self.map(features)
        features = features * mask
        for index in range(8, 13):
            features = getattr(b, f"block{index}")(features)
        features = b.relu(b.bn3(b.conv3(features)))
        features = b.bn4(b.conv4(features))
        logits = b.classifier(features)
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": features,
            "mask": mask,
            "vec": vector,
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]

    def loss_terms(self, outputs, labels, *, mask=None):
        if mask is not None:
            raise ValueError(
                "FFD mask supervision is unconnected: source loss requires mask_pred "
                "but forward emits mask; an explicit verified repair is required"
            )
        classification = F.cross_entropy(outputs["cls"], labels)
        return {"overall": classification, "cls": classification}
