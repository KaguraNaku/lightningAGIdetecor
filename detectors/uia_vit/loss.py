"""Native Mahalanobis patch consistency and shared scalar regularization."""

import torch
from torch import nn
from torch.nn import functional as F


def mahalanobis_distance(values, mean, precision):
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
    centered = values - mean
    return torch.einsum("im,mn,in->i", centered, precision, centered).sqrt()


class PatchConsistencyLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.c_real, self.c_fake, self.c_cross = (
            model.c_real,
            model.c_fake,
            model.c_cross,
        )

    def forward(
        self,
        real_attention,
        fake_attention,
        patches,
        real_mean,
        real_precision,
        fake_mean,
        fake_precision,
        labels,
    ):
        batch, height, width, channels = patches.shape
        real_distance = mahalanobis_distance(
            patches.reshape(batch * height * width, channels),
            real_mean.to(patches.device),
            real_precision.to(patches.device),
        )
        fake_distance = mahalanobis_distance(
            patches.reshape(batch * height * width, channels),
            fake_mean.to(patches.device),
            fake_precision.to(patches.device),
        )
        selected_fake = torch.where(labels == 1.0)[0]
        index_map = F.relu(real_distance - fake_distance).reshape(batch, height, width)[
            selected_fake, :
        ]
        if real_attention.shape[0] == 0:
            real_loss = 0
        else:
            count, positions, _ = real_attention.shape
            target = (1 - self.c_real) * torch.eye(
                positions, device=patches.device
            ) + self.c_real * torch.ones(positions, device=patches.device)
            target = target.expand(count, -1, -1)
            real_loss = torch.sum(torch.abs(real_attention - target)) / (
                count * (positions * positions - positions)
            )
        if fake_attention.shape[0] == 0:
            fake_loss = 0
        else:
            count, positions, _ = fake_attention.shape
            targets = []
            for index in range(count):
                fake = torch.where(index_map[index].reshape(-1) > 0)[0]
                real = torch.where(index_map[index].reshape(-1) <= 0)[0]
                target = (
                    torch.zeros(positions, positions, device=patches.device)
                    + self.c_cross
                )
                for position in fake:
                    target[position, fake] = self.c_fake
                for position in real:
                    target[position, real] = self.c_real
                targets.append(target)
            target = torch.stack(targets).to(patches.device)
            fake_loss = torch.sum(torch.abs(fake_attention - target)) / (
                count * (positions * positions - positions)
            )
        return real_loss + fake_loss
