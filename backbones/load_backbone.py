"""One selected-backbone loader, with no dependency imports at module load."""

from importlib import import_module
from pathlib import Path

from .registry import LOADERS


def get_backbone_model_type(model_id, **kwargs):
    """Read the Hugging Face architecture without loading model weights."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_id, **kwargs).model_type


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
            config_options = {key: value for key, value in kwargs.items() if key in self.CONFIG_OPTIONS}
            model_type = get_backbone_model_type(model_id, **config_options)
        loader = self.registry.resolve(model_type, component, importer=import_module)
        return loader(model_id, **kwargs)


_DEFAULT_LOADER = BackboneLoader()


def load_backbone(model_id, component="vision", *, model_type=None, **kwargs):
    """Compatibility API; imports and constructs only the selected component."""
    return _DEFAULT_LOADER.load(model_id, component, model_type=model_type, **kwargs)


def resolve_backbone_revision(model_id, revision=None):
    """Pin Hub files to one snapshot. Local model directories already select the files."""
    if Path(model_id).expanduser().is_dir():
        return None
    from huggingface_hub import resolve_revision

    return resolve_revision(model_id, revision=revision).resolved


def load_image_processor(model_id, *, revision=None, state=None):
    """Select the pretrained processor or restore its recorded configuration."""
    import transformers

    if state is None:
        return transformers.AutoImageProcessor.from_pretrained(model_id, revision=revision, backend="pil")

    from transformers.image_processing_utils import BaseImageProcessor

    name = state["image_processor_type"]
    processor_class = getattr(transformers, name, None)
    if not isinstance(processor_class, type) or not issubclass(processor_class, BaseImageProcessor):
        raise ValueError(f"Unknown image processor in checkpoint: {name}")
    return processor_class.from_dict(state)
