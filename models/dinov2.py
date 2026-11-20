"""DINOv2 vision loader; no Transformers model import before invocation."""


def load_dinov2_vision(model_id, **kwargs):
    from transformers import Dinov2Model

    return Dinov2Model.from_pretrained(model_id, **kwargs)
