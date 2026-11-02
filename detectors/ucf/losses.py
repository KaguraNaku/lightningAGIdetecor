"""Native shuffled-anchor contrastive regularization; no epsilon changes."""

from collections import defaultdict
import random

import torch
from torch import nn
from torch.nn import functional as F


def swap_specific(labels, values):
    labels = labels.cpu().numpy().tolist()
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    for indices in groups.values():
        random.shuffle(indices)
    return values[[groups[label].pop() for label in labels]]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=3.0):
        super().__init__()
        self.margin = margin

    def contrastive_loss(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        return torch.mean(
            torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)
        )

    def forward(self, common, specific, labels):
        batch = common.shape[0]
        real_common, fake_common = common.chunk(2)
        indices = list(range(batch // 2))
        random.shuffle(indices)
        real_anchor = common[indices]
        indices = list(range(batch // 2, batch))
        random.shuffle(indices)
        fake_anchor = common[indices]
        real_specific_anchor, fake_specific_anchor = swap_specific(
            labels, specific
        ).chunk(2)
        real_specific, fake_specific = specific.chunk(2)
        first = self.contrastive_loss(real_common, real_anchor, fake_anchor)
        second = self.contrastive_loss(fake_common, fake_anchor, real_anchor)
        third = self.contrastive_loss(
            real_specific, real_specific_anchor, fake_specific_anchor
        )
        fourth = self.contrastive_loss(
            fake_specific, fake_specific_anchor, real_specific_anchor
        )
        return first + second + fourth + third
