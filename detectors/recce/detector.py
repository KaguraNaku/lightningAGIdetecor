"""RECCE encoder/decoder reasoning with preserved native auxiliary observations."""

import torch
from torch import nn
from torch.nn import functional as F

from backbones.models.xception import Block, SeparableConv2d

from .reasoning import GraphReasoning, GuidedAttention


def decoder(inputs, outputs):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        SeparableConv2d(inputs, outputs, 3, 1, 1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
    )


class ReconstructionModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.loss_inputs = {}
        self.encoder = encoder
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2048, 2)
        self.attention = GuidedAttention()
        self.reasoning = GraphReasoning()
        self.decoder1 = decoder(728, 256)
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = decoder(256, 128)
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = decoder(128, 64)
        self.decoder6 = nn.Sequential(nn.Conv2d(64, 3, 1, bias=False), nn.Tanh())

    def norm_n_corr(self, features):
        normalized = F.normalize(self.global_pool(features), p=2, dim=1)
        # Preserve native batch-one squeeze behavior in the observed auxiliary.
        correlation = (
            torch.matmul(normalized.squeeze(), normalized.squeeze().T) + 1.0
        ) / 2.0
        return normalized, correlation

    @staticmethod
    def add_white_noise(images):
        gate = torch.rand([images.shape[0], 1, 1, 1])
        gate = torch.where(gate > 0.5, 1.0, 0.0).to(images.device)
        noise = torch.normal(0.0, 1e-6, size=images.shape, device=images.device)
        return torch.clip(images + noise * gate, -1.0, 1.0)

    def features(self, images):
        self.loss_inputs = {"recons": [], "contra": []}
        encoder = self.encoder
        inputs = self.add_white_noise(images) if self.training else images
        embedding = encoder.act1(encoder.bn1(encoder.conv1(inputs)))
        embedding = encoder.act2(encoder.bn2(encoder.conv2(embedding)))
        for index in range(1, 5):
            embedding = getattr(encoder, f"block{index}")(embedding)
        _, correlation = self.norm_n_corr(embedding)
        self.loss_inputs["contra"].append(correlation)
        decoded2 = self.decoder2(self.decoder1(self.dropout(embedding)))
        _, correlation = self.norm_n_corr(decoded2)
        self.loss_inputs["contra"].append(correlation)
        decoded4 = self.decoder4(self.decoder3(decoded2))
        _, correlation = self.norm_n_corr(decoded4)
        self.loss_inputs["contra"].append(correlation)
        reconstructed = self.decoder6(self.decoder5(decoded4))
        reconstructed = F.interpolate(
            reconstructed, size=images.shape[-2:], mode="bilinear", align_corners=True
        )
        self.loss_inputs["recons"].append(reconstructed)
        for index in (5, 6, 7):
            embedding = getattr(encoder, f"block{index}")(embedding)
        fusion = self.reasoning(embedding, decoded2, decoded4) + embedding
        embedding = encoder.block8(fusion)
        embedding = self.attention(images, reconstructed, embedding)
        for index in (9, 10, 11, 12):
            embedding = getattr(encoder, f"block{index}")(embedding)
        embedding = encoder.act3(encoder.bn3(encoder.conv3(embedding)))
        embedding = encoder.act4(encoder.bn4(encoder.conv4(embedding)))
        embedding = self.global_pool(embedding).squeeze(2).squeeze(2)
        return self.dropout(embedding)


class RECCEDetector(nn.Module):
    def __init__(self, backbone, encoder):
        super().__init__()
        self.backbone = (
            backbone  # Native unused wrapper branch remains in strict state.
        )
        self.model = ReconstructionModel(encoder)

    def forward_outputs(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("RECCE requires RGB images at native 256x256 geometry")
        features = self.model.features(images)
        logits = self.model.fc(features)
        return {
            "cls": logits,
            "feat": features,
            "reconstructed": self.model.loss_inputs["recons"][0],
            "correlations": self.model.loss_inputs["contra"],
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
