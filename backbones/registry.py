LOADERS = {
    ("clip", "vision"): ("backbones.models.clip", "load_clip_vision"),
    ("clip", "text"): ("backbones.models.clip", "load_clip_text"),
    ("clip", "full"): ("backbones.models.clip", "load_clip_model"),
}

def register_backbone(model_type, component:str):
    def register(loader):
        key = (model_type, component)
        if key in LOADERS:
            raise ValueError(f"Duplicate backbone: {key}")
        LOADERS[key] = loader
        return loader
    return register