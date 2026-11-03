LOADERS = {
    ("clip", "vision"): ("backbones.models.clip", "load_clip_vision"),
    ("clip", "text"): ("backbones.models.clip", "load_clip_text"),
    ("clip", "full"): ("backbones.models.clip", "load_clip_model"),
    ("clip_vision_model", "vision"): ("backbones.models.clip", "load_clip_vision"),
    ("dinov2", "vision"): ("backbones.models.dinov2", "load_dinov2_vision"),
    ("siglip", "vision"): ("backbones.models.siglip", "load_siglip_vision"),
    ("siglip_vision_model", "vision"): (
        "backbones.models.siglip",
        "load_siglip_vision",
    ),
}


def register_backbone(model_type, component: str):
    """Register an importable loader by path, without instantiating a model."""

    def register(loader):
        key = (model_type, component)
        if key in LOADERS:
            raise ValueError(f"Duplicate backbone: {key}")
        if loader.__qualname__ != loader.__name__:
            raise ValueError("a backbone loader must be importable at module scope")
        LOADERS[key] = (loader.__module__, loader.__name__)
        return loader

    return register
