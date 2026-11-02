"""Native detached-center RIL and its CE/late-stage composition."""

import torch
from torch import nn
from torch.nn import functional as F

from .attention import AttentionPooling


class RegionIndependentLoss(nn.Module):
    def __init__(self, decay_batch):
        super().__init__()
        self.register_buffer("feature_centers", torch.zeros(4, 32))
        self.register_buffer("intra_margin", torch.tensor([0.05, 0.1]))
        self.alpha, self.alpha_decay, self.decay_batch = 0.05, 0.9, decay_batch
        self.batch_cnt = 0
        self.atp = AttentionPooling()

    def forward(self, features, attentions, labels):
        batch = features.shape[0]
        if features.shape[-2:] != attentions.shape[-2:]:
            attentions = F.interpolate(
                attentions, features.shape[-2:], mode="bilinear", align_corners=True
            )
        matrix = self.atp(features, attentions)
        centers = self.feature_centers.detach()
        updated = (centers + self.alpha * torch.mean(matrix - centers, dim=0)).detach()
        with torch.no_grad():
            self.feature_centers = updated
        margins = torch.gather(
            self.intra_margin.repeat(batch, 1), 1, labels.unsqueeze(1)
        )
        intra = torch.mean(F.relu(torch.norm(matrix - updated, dim=-1) - margins))
        inter = 0
        for first in range(4):
            for second in range(first + 1, 4):
                inter += F.relu(
                    0.2 - torch.dist(updated[first], updated[second]), inplace=False
                )
        inter = inter / 4 / self.alpha
        self.batch_cnt += 1
        if self.batch_cnt % self.decay_batch == 0:
            self.alpha *= self.alpha_decay
        return inter + intra


class MultiAttentionObjective(nn.Module):
    def __init__(self, decay_batch):
        super().__init__()
        self.ril = RegionIndependentLoss(decay_batch)

    def forward(self, outputs, labels, *, late):
        ce = F.cross_entropy(outputs["cls"], labels)
        if not late:
            return ce
        return ce + self.ril(outputs["feature_maps_d"], outputs["attentions"], labels)
