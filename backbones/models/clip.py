def load_clip_vision(model_id, **kwargs):
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(model_id, **kwargs)


def load_clip_text(model_id, **kwargs):
    from transformers import CLIPTextModel

    return CLIPTextModel.from_pretrained(model_id, **kwargs)


def load_clip_model(model_id, **kwargs):
    from transformers import CLIPModel

    return CLIPModel.from_pretrained(model_id, **kwargs)
