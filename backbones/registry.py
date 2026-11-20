"""Lazy loaders for foundation models and detector backbones."""

from collections.abc import MutableMapping
from importlib import import_module


class BackboneRegistry(MutableMapping):
    """Own loader declarations without importing or constructing model backends.

    A declaration is either an import path pair or an explicitly supplied
    callable. Resolving one declaration never imports the other backbones.
    """

    def __init__(self, entries=None):
        self._entries = dict(entries or {})

    def __getitem__(self, key):
        return self._entries[key]

    def __setitem__(self, key, value):
        self._entries[key] = value

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def register(self, model_type, component):
        key = (model_type, component)

        def decorate(loader):
            if key in self:
                raise ValueError(f"Duplicate backbone: {key}")
            if not callable(loader):
                raise TypeError("Backbone loader must be callable")
            self[key] = loader
            return loader

        return decorate

    def resolve(self, model_type, component, *, importer=import_module):
        key = (model_type, component)
        if key not in self:
            raise ValueError(f"Unsupported backbone component: {key}")
        entry = self[key]
        if callable(entry):
            return entry
        module_name, function_name = entry
        return getattr(importer(module_name), function_name)


LOADERS = BackboneRegistry({
    ("xception", "vision"): ("detectors.xception.backbone", "load_xception"),
    ("ucf_xception", "vision"): ("detectors.ucf.backbone", "load_ucf_xception"),
    ("efficientnetb4", "vision"): ("detectors.prodet.backbone", "load_efficientnetb4"),
    ("gend_clip", "vision"): ("detectors.gend.encoder", "load_gend_clip"),
    ("forada_clip", "full"): ("detectors.forada.clip", "load_forada_clip"),
    ("clip", "vision"): ("models.clip", "load_clip_vision"),
    ("clip_vision_model", "vision"): ("models.clip", "load_clip_vision"),
    ("dinov2", "vision"): ("models.dinov2", "load_dinov2_vision"),
    ("siglip", "vision"): ("models.siglip", "load_siglip_vision"),
    ("siglip_vision_model", "vision"): ("models.siglip", "load_siglip_vision"),
    ("vit", "vision"): ("models.vit", "load_vit_vision"),
})


def register_backbone(model_type, component):
    """Compatibility decorator backed by the common registry object."""
    return LOADERS.register(model_type, component)
