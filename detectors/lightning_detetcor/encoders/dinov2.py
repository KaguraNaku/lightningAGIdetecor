from .base import FrozenVisionEncoder


class DINOv2VisionEncoder(FrozenVisionEncoder):
    MODEL_TYPE = "dinov2"
    PREFIX_TOKENS = 1

    def _global(self, block):
        # Per-block CLS, not the final model-level post-normalized pooler.
        return block[:, 0]
