"""Backbone loaders: lazy (module, function) paths or registered callables.

Only the selected path is imported by load_backbone; decorator registration
stores an already available callable without importing other backends.
"""

LOADERS = {
    ("efficientnetb4", "vision"): ("backbones.models.efficientnetb4", "load_efficientnetb4"),
    ("resnet34", "vision"): ("backbones.models.resnet34", "load_resnet34"),
    ("meso4", "vision"): ("backbones.models.meso", "load_meso4"),
    ("meso4Inception", "vision"): ("backbones.models.meso", "load_meso_inception4"),
    ("xception", "vision"): ("backbones.models.xception", "load_xception"),
    ("clip", "vision"): ("backbones.models.clip", "load_clip_vision"),
    ("clip_vision_model", "vision"): ("backbones.models.clip", "load_clip_vision"),
    ("clip", "text"): ("backbones.models.clip", "load_clip_text"),
    ("clip", "full"): ("backbones.models.clip", "load_clip_model"),
}

def register_backbone(model_type, component:str):
    """Register a loader callable and reject duplicate model/component pairs."""
    def register(loader):
        key = (model_type, component)
        if key in LOADERS:
            raise ValueError(f"Duplicate backbone: {key}")
        LOADERS[key] = loader
        return loader
    return register
