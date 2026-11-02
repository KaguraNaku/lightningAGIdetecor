"""Trainable implicit/explicit identity branches with native checkpoint names."""

from torch import nn
from torch.nn import functional as F

from .objectives import BinaryMarginLoss, IdentityArcFaceLoss, l2_norm


class IIDDetector(nn.Module):
    def __init__(self, backbone, explicit_extractor):
        super().__init__()
        self.backbone = backbone
        self.loss_func = nn.CrossEntropyLoss()
        self.explicit_extractor = explicit_extractor
        self.explicit_extractor.eval()  # Constructor mode only; parent train() resets it.
        self.BCE_LOSS = BinaryMarginLoss()
        self.IIE_LOSS = IdentityArcFaceLoss()
        self.IIE_LOSS.train()

    def forward_outputs(self, images):
        identity = self.explicit_extractor(
            F.interpolate(images, size=(112, 112), mode="bilinear", align_corners=False)
        )
        features = self.backbone.features(images)
        logits = self.backbone.classifier(features, identity)
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": features,
            "id_feat": identity,
            "embed": self.backbone.last_emb,
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]

    def loss_terms(self, outputs, labels, *, id_indices=None):
        implicit, explicit = l2_norm(outputs["embed"]), l2_norm(outputs["id_feat"])
        positive, negative = labels == 1, labels == 0
        eic = (implicit[positive] * explicit[positive]).sum(dim=1).mean() - (
            implicit[negative] * explicit[negative]
        ).sum(dim=1).mean()
        binary = self.BCE_LOSS(outputs["cls"], labels, return_logits=True).mean()
        loss = 0
        loss += binary
        identity = 0
        if id_indices is not None:
            identity, _, _ = self.IIE_LOSS(
                outputs["embed"], id_indices, return_logits=True
            )
            loss += 0.05 * identity
        loss += 0.1 * eic
        return {
            "overall": loss,
            "loss_bce": binary,
            "loss_iie": identity,
            "loss_eic": eic,
        }
