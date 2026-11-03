"""Native VideoMAE token mean, second normalization and two-logit CE head."""

import torch
from torch import nn


class VideoMAEDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 2)
        self.loss_func = nn.CrossEntropyLoss()

    def classify_tokens(self, tokens):
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (1568, 768):
            raise ValueError("VideoMAE-base head needs N x1568 x768 unmasked tokens")
        features = self.fc_norm(tokens.mean(1))
        logits = self.head(features)
        return {"cls": logits, "prob": torch.softmax(logits, dim=1)[:, 1], "feat": features}

    def forward_outputs(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (16, 3, 224, 224):
            raise ValueError("VideoMAE-base needs NTCHW RGB224/T16 clips")
        return self.classify_tokens(self.backbone(clips))

    def forward(self, clips):
        return self.forward_outputs(clips)["cls"]

    def loss_terms(self, outputs, labels):
        if labels.shape != (outputs["cls"].shape[0],) or labels.dtype != torch.int64:
            raise ValueError("VideoMAE needs one int64 binary class per clip")
        if not labels.is_meta and not ((labels == 0) | (labels == 1)).all():
            raise ValueError("VideoMAE labels must be binary")
        return {"overall": self.loss_func(outputs["cls"], labels)}
