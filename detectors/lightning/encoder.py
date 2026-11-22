"""Frozen vision transformer readouts from selected blocks."""
import torch
from torch import nn
from torch.nn import functional as F

from . import config


GLOBAL_READOUTS = {
    "clip": "cls",
    "clip_vision_model": "cls",
    "dinov2": "cls",
    "vit": "cls",
    "siglip": "global",
    "siglip_vision_model": "global",
}


def global_readout_for(model_type):
    try:
        return GLOBAL_READOUTS[model_type]
    except KeyError:
        supported = ", ".join(GLOBAL_READOUTS)
        raise ValueError(
            f"Lightning does not support backbone architecture {model_type!r}. Supported: {supported}"
        ) from None


class VisionEncoder(nn.Module):
    def __init__(self, vision, layers=config.LAYERS):
        super().__init__()
        self.layers = tuple(layers)
        vision_config = vision.config
        self.global_branch = global_readout_for(vision_config.model_type)
        if max(self.layers) >= vision_config.num_hidden_layers:
            raise ValueError(f"Selected layers exceed the backbone's {vision_config.num_hidden_layers} blocks")
        self.width = vision_config.hidden_size
        self.uses_attention_pool = self.global_branch == "global"
        if self.uses_attention_pool and not hasattr(vision, "head"):
            raise ValueError("SigLIP requires its pretrained attention pooling head")
        self.patch_start = 0 if self.uses_attention_pool else 1 + getattr(vision_config, "num_register_tokens", 0)
        self.vision = vision.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        hidden = self.vision(pixel_values=images, output_hidden_states=True).hidden_states[1:]
        tokens = [hidden[i] for i in self.layers]
        global_features = (
            [self.vision.head(self.vision.post_layernorm(layer)) for layer in tokens]
            if self.uses_attention_pool else [layer[:, 0] for layer in tokens]
        )
        patch_means = [layer[:, self.patch_start:].mean(1) for layer in tokens]
        return {
            self.global_branch: torch.stack([F.normalize(values, dim=-1) for values in global_features], dim=1),
            "mean": torch.stack([F.normalize(values, dim=-1) for values in patch_means], dim=1),
        }
