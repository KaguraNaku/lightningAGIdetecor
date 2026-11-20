"""CLIP feature composition retained by the local GenD CLIP adapter."""

from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel


class GenDCLIPEncoder(nn.Module):
    def __init__(self, vision_config, *, attention_implementation="sdpa"):
        super().__init__()
        config = CLIPVisionConfig.from_dict(vision_config)
        config._attn_implementation = attention_implementation
        vision = CLIPVisionModel(config)
        self.vision_model = getattr(vision, "vision_model", vision)
        self.visual_projection = nn.Linear(
            config.hidden_size, config.projection_dim, bias=False
        )
        self.features_dim = config.hidden_size

    def forward(self, images):
        return self.vision_model(images).pooler_output


def load_gend_clip(model_id, **kwargs):
    return GenDCLIPEncoder(**kwargs)
