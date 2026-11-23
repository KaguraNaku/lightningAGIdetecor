"""Model composition only: CLIP vision, attention residuals and a binary head."""

from torch import nn

from .svd import replace_attention


class EffortDetector(nn.Module):
    def __init__(self, vision, *, retained_rank=None, initialize=True):
        super().__init__()
        # Store the vision transformer directly to preserve author checkpoint keys.
        vision = vision.vision_model if hasattr(vision, "vision_model") else vision
        self.retained_rank = (
            vision.config.hidden_size - 1 if retained_rank is None else retained_rank
        )
        self.backbone = replace_attention(
            vision, self.retained_rank, initialize=initialize
        )
        self.head = nn.Linear(vision.config.hidden_size, 2)

    def forward(self, images):
        return self.head(self.backbone(pixel_values=images).pooler_output)
