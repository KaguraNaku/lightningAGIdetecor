"""Canonical GOANet composition and its active CE/Gram objective."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import PriorContext, PriorGuidedGOALayer, replace_attention
from .prior import PriorExtractionBranch


class GOANetDetector(nn.Module):
    def __init__(self, backbone_factory):
        super().__init__()
        self.prior_branch = PriorExtractionBranch()
        self.context = PriorContext()
        self.backbone = replace_attention(backbone_factory(), self.context)
        self.head = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward_outputs(self, images):
        prior = self.prior_branch(images)
        with self.context.bind(prior):
            features = self.backbone(images).pooler_output
        logits = self.head(features)
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": features,
            "prior_feat": prior,
            "feat_pri_list": [],
            "feat_res_list": [],
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]

    def loss_terms(self, outputs, labels):
        logits = outputs["cls"]
        classification = F.cross_entropy(logits, labels)
        orthogonal = torch.tensor(0.0, device=logits.device)
        for module in self.backbone.modules():
            if isinstance(module, PriorGuidedGOALayer):
                orthogonal += module.compute_orthogonal_loss()
        real, fake = labels == 0, labels == 1
        return {
            "overall": classification + 0.01 * orthogonal,
            "cls": classification,
            "ortho": orthogonal,
            "consistency": torch.tensor(0.0, device=logits.device),
            "real_loss": F.cross_entropy(logits[real], labels[real])
            if real.sum() > 0
            else torch.tensor(0.0, device=logits.device),
            "fake_loss": F.cross_entropy(logits[fake], labels[fake])
            if fake.sum() > 0
            else torch.tensor(0.0, device=logits.device),
        }
