"""Selected CLIP vision component; importing this module loads no model library."""


def load_clip_vision(model_id, **kwargs):
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(model_id, **kwargs)
