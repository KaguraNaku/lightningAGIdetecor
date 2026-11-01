from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel

def load_clip_vision(model_id, **kwargs):
    return CLIPVisionModel.from_pretrained(model_id, **kwargs)

def load_clip_text(model_id, **kwargs):
    return CLIPTextModel.from_pretrained(model_id, **kwargs)

def load_clip_model(model_id, **kwargs):
    return CLIPModel.from_pretrained(model_id, **kwargs)
