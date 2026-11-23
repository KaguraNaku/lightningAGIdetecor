"""Equal raw-score fusion of the two analytic Lightning branches."""

import torch
from torch import nn


class LightningDetector(nn.Module):
    def __init__(self, backbone, coefficients):
        super().__init__()
        self.backbone = backbone
        self.branches = tuple(key for key in coefficients if key != "bias")
        for branch in self.branches:
            self.register_buffer(
                branch, torch.as_tensor(coefficients[branch], dtype=torch.float64)
            )
        self.register_buffer(
            "bias", torch.as_tensor(coefficients["bias"], dtype=torch.float64)
        )

    @torch.inference_mode()
    def forward(self, images):
        features = self.backbone(images)
        result = self.bias.expand(len(images)).clone()
        for branch in self.branches:
            result += (
                features[branch].double().flatten(1) @ getattr(self, branch).flatten()
            )
        return result
