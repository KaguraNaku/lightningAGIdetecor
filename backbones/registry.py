"""Lazy backbone registry for the nine supported detector tasks."""

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
    ("xception", "vision"): ("backbones.models.xception", "load_xception"),
    ("ucf_xception", "vision"): ("backbones.models.ucf_xception", "load_ucf_xception"),
    ("efficientnetb4", "vision"): ("backbones.models.efficientnetb4", "load_efficientnetb4"),
    ("gend_clip", "vision"): ("backbones.models.gend_clip", "load_gend_clip"),
    ("forada_clip", "full"): ("backbones.models.forada_clip", "load_forada_clip"),
    ("clip", "vision"): ("backbones.models.clip", "load_clip_vision"),
    ("clip_vision_model", "vision"): ("backbones.models.clip", "load_clip_vision"),
})


def register_backbone(model_type, component):
    """Compatibility decorator backed by the common registry object."""
    return LOADERS.register(model_type, component)
