"""Local per-position graph reasoning and reconstruction-guided attention."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphReasoning(nn.Module):
    def __init__(self):
        super().__init__()
        self.ratio = [2, 4]
        self.va_embedding = nn.Sequential(
            nn.Conv2d(728, 256, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1, bias=False),
        )
        self.va_gated_b = nn.Sequential(
            nn.Conv2d(728, 256, 1, bias=False), nn.Sigmoid()
        )
        self.va_gated_c = nn.Sequential(
            nn.Conv2d(728, 256, 1, bias=False), nn.Sigmoid()
        )
        self.vb_embedding = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(True),
            nn.Linear(256, 256, bias=False),
        )
        self.vc_embedding = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.ReLU(True),
            nn.Linear(256, 256, bias=False),
        )
        self.unfold_b = nn.Unfold(kernel_size=2, stride=2)
        self.unfold_c = nn.Unfold(kernel_size=4, stride=4)
        self.reweight_ab = nn.Sequential(
            nn.Linear(512, 1, bias=False), nn.ReLU(True), nn.Softmax(dim=1)
        )
        self.reweight_ac = nn.Sequential(
            nn.Linear(512, 1, bias=False), nn.ReLU(True), nn.Softmax(dim=1)
        )
        self.reproject = nn.Sequential(
            nn.Conv2d(768, 728, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(728, 728, 1, bias=False),
            nn.Dropout(0.2),
        )

    def forward(self, a, b, c):
        embedded = self.va_embedding(a)
        embedded = embedded.reshape(embedded.shape[0], embedded.shape[1], -1)
        gate_b = (1 - self.va_gated_b(a)).reshape(*embedded.shape)
        gate_c = (1 - self.va_gated_c(a)).reshape(*embedded.shape)
        neighbors_b = (
            self.unfold_b(b).reshape(b.shape[0], b.shape[1], 4, -1).permute(0, 2, 3, 1)
        )
        neighbors_b = self.vb_embedding(neighbors_b)
        neighbors_c = (
            self.unfold_c(c).reshape(c.shape[0], c.shape[1], 16, -1).permute(0, 2, 3, 1)
        )
        neighbors_c = self.vc_embedding(neighbors_c)
        aggregated_b, aggregated_c = [], []
        for position in range(embedded.shape[-1]):
            repeated = torch.stack([embedded[:, :, position]] * 4, dim=1)
            neighbor = neighbors_b[:, :, position, :]
            weights = self.reweight_ab(torch.cat([repeated, neighbor], dim=-1))
            aggregated_b.append(
                torch.bmm(neighbor.transpose(1, 2), weights).squeeze()
                * gate_b[:, :, position]
            )
            repeated = torch.stack([embedded[:, :, position]] * 16, dim=1)
            neighbor = neighbors_c[:, :, position, :]
            weights = self.reweight_ac(torch.cat([repeated, neighbor], dim=-1))
            aggregated_c.append(
                torch.bmm(neighbor.transpose(1, 2), weights).squeeze()
                * gate_c[:, :, position]
            )
        aggregates = torch.cat(
            [torch.stack(aggregated_b, dim=-1), torch.stack(aggregated_c, dim=-1)],
            dim=1,
        )
        fused = torch.sigmoid(torch.cat([aggregates, embedded], dim=1))
        return self.reproject(fused.reshape(a.shape[0], -1, a.shape[2], a.shape[3]))


class GuidedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.h = nn.Sequential(
            nn.Conv2d(728, 728, 1, bias=False), nn.BatchNorm2d(728), nn.ReLU(True)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, images, reconstructed, embedding):
        residual = F.interpolate(
            torch.abs(images - reconstructed),
            size=embedding.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        weights = self.gated(residual)
        return weights * self.h(embedding) + self.dropout(embedding)
