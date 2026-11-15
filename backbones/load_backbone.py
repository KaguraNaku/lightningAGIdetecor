"""One selected-backbone loader, with no dependency imports at module load."""

from importlib import import_module

from .registry import LOADERS


class BackboneLoader:
    """Resolve native or Hugging Face model types only when loading is requested."""

    CONFIG_OPTIONS = frozenset({
        "revision", "cache_dir", "local_files_only", "token", "force_download",
        "subfolder", "trust_remote_code",
    })

    def __init__(self, registry=None):
        self.registry = LOADERS if registry is None else registry

    def load(self, model_id, component="vision", *, model_type=None, **kwargs):
        if model_type is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_id,
                **{key: value for key, value in kwargs.items() if key in self.CONFIG_OPTIONS},
            )
            model_type = config.model_type
        loader = self.registry.resolve(model_type, component, importer=import_module)
        return loader(model_id, **kwargs)


_DEFAULT_LOADER = BackboneLoader()


def load_backbone(model_id, component="vision", *, model_type=None, **kwargs):
    """Compatibility API; imports and constructs only the selected component."""
    return _DEFAULT_LOADER.load(model_id, component, model_type=model_type, **kwargs)
