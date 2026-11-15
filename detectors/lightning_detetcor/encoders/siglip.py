from .base import FrozenVisionEncoder


class SigLIPVisionEncoder(FrozenVisionEncoder):
    """SigLIP and fixed-resolution SigLIP2; these models have no CLS token."""

    MODEL_TYPE = "siglip_vision_model"

    def __init__(self, vision_model, model_id=None, *, readout="global", revision=None):
        super().__init__(vision_model, model_id, readout=readout, revision=revision)
        if not hasattr(self._tower, "head") or not hasattr(
            self._tower, "post_layernorm"
        ):
            raise ValueError(
                "SigLIP readouts require the checkpoint's native pooling head"
            )

    @property
    def _tower(self):
        # Transformers has exposed both nested and flat vision-model layouts.
        return getattr(self.model, "vision_model", self.model)

    def _global(self, block):
        return self._tower.head(self._tower.post_layernorm(block))

    def _model_kwargs(self):
        # Fixed checkpoint resolution; interpolation is not enabled implicitly.
        return {"interpolate_pos_encoding": False}

    def _validate_pixel_values(self, pixel_values):
        super()._validate_pixel_values(pixel_values)
        size = self.model.config.image_size
        expected = (size, size) if isinstance(size, int) else tuple(size)
        if tuple(pixel_values.shape[-2:]) != expected:
            raise ValueError(
                f"SigLIP requires checkpoint-processor resolution {expected}"
            )
