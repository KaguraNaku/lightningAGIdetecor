"""Native phase reconstruction stays in the model and follows RGB normalization."""

import torch
from torch import nn


class SPSLDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def phase_without_amplitude(images):
        gray = images.mean(dim=1, keepdim=True)
        spectrum = torch.fft.fftn(gray, dim=(-1, -2))
        phase = torch.angle(spectrum)
        reconstructed = torch.exp(1j * phase)
        return torch.real(torch.fft.ifftn(reconstructed, dim=(-1, -2)))

    def forward_outputs(self, images):
        phase = self.phase_without_amplitude(images)
        features = self.backbone.features(torch.cat((images, phase), dim=1))
        logits = self.backbone.classifier(features)
        return {"cls": logits, "feat": features, "phase": phase}

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
