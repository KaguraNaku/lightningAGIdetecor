"""Equal raw-score fusion of the two analytic Lightning branches."""
import torch
from torch import nn

from .model import BRANCHES


class LightningDetector(nn.Module):
    def __init__(self, backbone, coefficients):
        super().__init__()
        self.backbone = backbone
        for branch in BRANCHES:
            self.register_buffer(branch, torch.as_tensor(coefficients[branch], dtype=torch.float64))
        self.register_buffer('bias', torch.as_tensor(coefficients['bias'], dtype=torch.float64))

    @torch.inference_mode()
    def forward(self, images):
        features = self.backbone(images)
        result = self.bias.expand(len(images)).clone()
        for branch in BRANCHES:
            result += features[branch].double().flatten(1) @ getattr(self, branch).flatten()
        return result
