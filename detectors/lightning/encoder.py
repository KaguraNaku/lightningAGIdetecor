"""Frozen vision transformer readouts from selected blocks."""
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_BACKBONE_ID = "openai/clip-vit-large-patch14"
DEFAULT_LAYERS = (0, 3, 7, 10, 13, 16, 20, 23)


class VisionEncoder(nn.Module):
    def __init__(self, vision, layers=DEFAULT_LAYERS):
        super().__init__()
        self.vision = vision.eval().requires_grad_(False)
        self.layers = tuple(layers)
        config = vision.config
        if max(self.layers) >= config.num_hidden_layers:
            raise ValueError(f"Selected layers exceed the backbone's {config.num_hidden_layers} blocks")
        self.width = config.hidden_size
        self.uses_attention_pool = config.model_type in ("siglip", "siglip_vision_model")
        if self.uses_attention_pool and not hasattr(vision, "head"):
            raise ValueError("SigLIP requires its pretrained attention pooling head")
        self.global_branch = "global" if self.uses_attention_pool else "cls"
        self.patch_start = 0 if self.uses_attention_pool else 1 + getattr(config, "num_register_tokens", 0)

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
