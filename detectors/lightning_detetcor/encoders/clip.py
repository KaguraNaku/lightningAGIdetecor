from .base import FrozenVisionEncoder


class CLIPReadoutEncoder(FrozenVisionEncoder):
    """Optional patch readouts; the legacy CLIP global class stays unchanged."""

    MODEL_TYPE = "clip_vision_model"
    PREFIX_TOKENS = 1

    def _global(self, block):
        return block[:, 0]
