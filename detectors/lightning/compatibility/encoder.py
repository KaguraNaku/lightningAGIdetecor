"""Lightning-specific CLIP readout: per-block CLS features with L2 normalization."""

import torch
from torch import nn
from torch.nn import functional as F


class CLIPVisionEncoder(nn.Module):
    def __init__(self, vision_model, model_id=None):
        super().__init__()
        if vision_model.config.model_type != "clip_vision_model":
            raise ValueError("CLIPVisionEncoder requires CLIPVisionModel")
        self.model = vision_model
        self.model_id = (
            model_id or vision_model.config._name_or_path or "clip_vision_model"
        )
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # This adapter always uses the frozen backbone in evaluation mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values):
        """Return per-block L2-normalized CLS features from processed images."""
        parameter = next(self.model.parameters())
        outputs = self.model(
            pixel_values=pixel_values.to(parameter),
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] is the embedding output, before transformer blocks.
        features = torch.stack(
            [hidden[:, 0, :] for hidden in outputs.hidden_states[1:]], dim=1
        )
        return F.normalize(features, dim=-1)
