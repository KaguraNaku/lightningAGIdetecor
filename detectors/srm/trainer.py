"""Native raw-logit cosine-margin objective; inference remains ordinary softmax."""

import torch
from torch import nn
from torch.nn import functional as F

from detectors._shared.trainer import ImageClassificationTrainer


class SRMMarginLoss(nn.Module):
    def forward(self, logits, labels):
        selected = torch.zeros_like(logits, dtype=torch.bool)
        selected.scatter_(1, labels.view(-1, 1), True)
        output = torch.where(selected, logits - 0.45, logits)
        return F.cross_entropy(30 * output, labels)


class SRMTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=SRMMarginLoss())
