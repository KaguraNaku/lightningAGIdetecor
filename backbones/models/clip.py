from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel

def load_clip_vision(model_id):
    return CLIPVisionModel.from_pretrained(model_id)

def load_clip_text(model_id):
    return CLIPTextModel.from_pretrained(model_id)

def load_clip_model(model_id):
    return CLIPModel.from_pretrained(model_id)