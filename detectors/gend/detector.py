"""Explicit GenD composition; unused visual projection remains checkpoint state."""

import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, dimension, *, normalize):
        super().__init__()
        self.linear = nn.Linear(dimension, 2)
        self.normalize_inputs = normalize

    def forward(self, features):
        if self.normalize_inputs:
            features = F.normalize(features, p=2, dim=1)
        return self.linear(features)


class GenDModel(nn.Module):
    def __init__(self, encoder, *, head):
        super().__init__()
        if head not in ("linear", "LinearNorm"):
            raise ValueError("GenD head must be linear or LinearNorm")
        self.feature_extractor = encoder
        self.model = LinearProbe(encoder.features_dim, normalize=head == "LinearNorm")

    def forward(self, images):
        return self.model(self.feature_extractor(images))


class GenDDetector(nn.Module):
    def __init__(self, encoder, *, head):
        super().__init__()
        self.model = GenDModel(encoder, head=head)
        for name, parameter in self.model.feature_extractor.named_parameters():
            parameter.requires_grad_("norm" in name.lower() or "ln" in name.lower())
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(True)

    def forward(self, images):
        return self.forward_outputs(images)["cls"]

    def forward_outputs(self, images):
        # The source explicitly disables CUDA AMP for the complete GenD model.
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.model.feature_extractor(images)
            embeddings = F.normalize(features, p=2, dim=1)
            logits = self.model.model.linear(embeddings if self.model.model.normalize_inputs else features)
            return {"cls": logits, "l2_embeddings": embeddings}
