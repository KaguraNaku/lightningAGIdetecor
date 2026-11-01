"""Backbone loaders: lazy (module, function) paths or registered callables.

Only the selected path is imported by load_backbone; decorator registration
stores an already available callable without importing other backends.
"""

LOADERS = {
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
