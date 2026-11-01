"""Frozen CLIP vision with trainable LoRA projections and a binary head."""

from torch import nn

from .layers import replace_attention


class LoRADetector(nn.Module):
    def __init__(self, vision, *, rank=16, alpha=16):
        super().__init__()
        vision = getattr(vision, "vision_model", vision)
        self.backbone = replace_attention(vision, rank=rank, alpha=alpha)
        parameter = next(vision.parameters())
        self.head = nn.Linear(
            vision.config.hidden_size, 2, device=parameter.device, dtype=parameter.dtype
        )

    def forward(self, images):
        return self.head(self.backbone(pixel_values=images).pooler_output)
