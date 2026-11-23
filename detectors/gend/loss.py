"""GenD CE + label alignment + hyperspherical uniformity, author revision a31acda."""

import torch
from torch import nn
from torch.nn import functional as F


class GenDLoss(nn.Module):
    def terms(self, outputs, labels):
        z = outputs["l2_embeddings"]
        if z.ndim != 2 or len(z) != len(labels) or len(z) < 2:
            raise ValueError(
                "GenD uniformity needs at least two aligned examples per batch"
            )
        ce = F.cross_entropy(outputs["cls"], labels)
        positive = torch.nonzero(
            (labels[:, None] == labels[None, :]).triu(1), as_tuple=False
        )
        alignment = (
            (z[positive[:, 0]] - z[positive[:, 1]]).norm(p=2, dim=1).pow(2).mean()
            if len(positive)
            else z.sum() * 0
        )
        uniformity = (
            torch.pdist(z, p=2).pow(2).mul(-2).exp().mean().clamp(min=1e-6).log()
        )
        total = ce + 0.1 * alignment + 0.5 * uniformity
        if not torch.isfinite(total):
            raise ValueError(
                "Nonfinite GenD objective; refusing the upstream NaN-to-zero fallback"
            )
        return {
            "ce": ce,
            "alignment": alignment,
            "uniformity": uniformity,
            "overall": total,
        }

    def forward(self, outputs, labels):
        return self.terms(outputs, labels)["overall"]
