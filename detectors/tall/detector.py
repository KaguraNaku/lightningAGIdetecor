# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright(c)2023 CUHK(SZ),Kangran Zhao;adapted from DeepfakeBench.
"""TALL native packed-T4 input,two-logit CE and probability feature alias."""

import torch
from torch import nn


class TALLDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (4, 3, 224, 224):
            raise ValueError("TALL requires NTCHW T4/RGB224 clips")
        bs, t, c, h, w = clips.shape
        return self.model(clips.view(bs, t * c, h, w))

    def forward_outputs(self, clips):
        logits = self(clips)
        probability = logits.softmax(1)[:, 1]
        return {"cls": logits, "prob": probability, "feat": probability}

    def loss_terms(self, outputs, labels):
        if labels.dtype != torch.int64 or labels.shape != (len(outputs["cls"]),):
            raise ValueError("TALL needs one int64 binary label per clip")
        if not labels.is_meta and not ((labels == 0) | (labels == 1)).all():
            raise ValueError("TALL labels must be binary")
        return {"overall": self.loss_func(outputs["cls"], labels.long())}
