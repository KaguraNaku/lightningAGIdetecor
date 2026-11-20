"""Frozen CLIP per-block CLS and patch-mean readouts."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import LAYERS


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_WIDTH = 1024


class CLIPEncoder(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        hidden = self.vision(pixel_values=images, output_hidden_states=True).hidden_states[1:]
        if len(hidden) != 24:
            raise ValueError("The formal fixed-eight protocol requires 24 CLIP blocks")
        return {
            'cls': torch.stack([F.normalize(hidden[i][:, 0], dim=-1) for i in LAYERS], 1),
            'mean': torch.stack([F.normalize(hidden[i][:, 1:].mean(1), dim=-1) for i in LAYERS], 1),
        }
