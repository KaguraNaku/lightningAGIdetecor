"""CLIP pooled vision features and a fully trainable binary classifier."""

from torch import nn


class CLIPDetector(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.backbone = getattr(vision, "vision_model", vision)
        parameter = next(self.backbone.parameters())
        self.head = nn.Linear(
            self.backbone.config.hidden_size,
            2,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def features(self, images):
        return self.backbone(pixel_values=images).pooler_output

    def forward(self, images):
        return self.head(self.features(images))
