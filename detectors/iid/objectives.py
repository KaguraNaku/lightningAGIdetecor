"""Literal native sigmoid-margin binary loss, ArcFace1000 and signed EIC."""

import math

import torch
from torch import nn


def l2_norm(value, axis=1):
    return torch.div(value, torch.norm(value, p=2, dim=axis, keepdim=True))


class BinaryMarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = nn.CrossEntropyLoss(reduction="mean")
        self.sig = nn.Sigmoid()

    def forward(self, embeddings, labels, return_logits=False):
        probabilities = self.sig(embeddings)
        target = (
            probabilities[
                torch.arange(embeddings.size(0), device=labels.device), labels
            ].view(-1, 1)
            - 0.2
        )
        output = probabilities.clone()
        output.scatter_(1, labels.view(-1, 1).long(), target)
        return self.criteria(output * 8.0, labels)


class IdentityArcFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(512, 1000))
        nn.init.normal_(self.kernel, std=0.01)
        self.criteria = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, embeddings, labels, return_logits=False):
        cos_theta = torch.mm(
            l2_norm(embeddings, axis=1), l2_norm(self.kernel, axis=0)
        ).clamp(-1, 1)
        with torch.no_grad():
            original = cos_theta.clone()
        target = cos_theta[
            torch.arange(embeddings.size(0), device=labels.device), labels
        ].view(-1, 1)
        sine = torch.sqrt(1.0 - torch.pow(target, 2))
        adjusted = target * math.cos(0.4) - sine * math.sin(0.4)
        margin = torch.where(
            target > math.cos(math.pi - 0.4),
            adjusted,
            target - math.sin(math.pi - 0.4) * 0.4,
        )
        cos_theta.scatter_(1, labels.view(-1, 1).long(), margin)
        output = cos_theta * 64.0
        loss = self.criteria(output, labels)
        with torch.no_grad():
            _, selected = torch.max(original * 64.0, dim=1, keepdim=False)
            accuracy = (selected == labels).sum().item() / labels.size(0)
        return (loss, accuracy, output) if return_logits else (loss, accuracy)
