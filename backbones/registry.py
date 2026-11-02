"""Backbone loaders: lazy (module, function) paths or registered callables.

Only the selected path is imported by load_backbone; decorator registration
stores an already available callable without importing other backends.
"""

LOADERS = {
    ("hrnet_w48", "vision"): ("backbones.models.hrnet_w48", "load_hrnet_w48"),
    ("clip_checkpoint", "vision"): ("backbones.models.clip_checkpoint", "load_clip_checkpoint"),
    ("effluxlet_clip", "vision"): ("backbones.models.effluxlet_clip", "load_effluxlet_clip"),
    ("gend_clip", "vision"): ("backbones.models.gend_clip", "load_gend_clip"),
    ("vgg19_capsule", "vision"): ("backbones.models.vgg19_capsule", "load_vgg19_capsule"),
    ("uia_vit", "vision"): ("backbones.models.uia_vit", "load_uia_vit"),
    ("xception_timm", "vision"): ("backbones.models.xception_timm", "load_xception_encoder"),
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
