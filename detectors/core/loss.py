"""Native consistency cosine MSE plus CE, with positional batch-halves pairing."""

import torch
from torch import nn
from torch.nn import functional as F


class ConsistencyCosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.mse_fn = nn.MSELoss()

    def forward(self, features, logits, labels):
        if features.ndim != 2 or len(features) != len(labels) or len(labels) < 2:
            raise ValueError(
                "CORE consistency requires aligned features and at least two samples per batch"
            )
        features = F.normalize(features, dim=1)
        half = len(features) // 2
        cosine = torch.einsum(
            "nc,nc->n", [features[:half], features[half : 2 * half]]
        ).unsqueeze(-1)
        targets = torch.ones(
            (len(cosine), 1), dtype=torch.float32, device=cosine.device
        )
        return 1.0 * self.mse_fn(cosine, targets) + self.loss_fn(logits, labels)
