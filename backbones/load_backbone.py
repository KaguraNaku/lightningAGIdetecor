from importlib import import_module

from .registry import LOADERS


def load_backbone(model_id: str, component: str = "vision", **kwargs):
    """Import and load only the requested component; never move it to a GPU.

    Factory/registry imports do not import Transformers or model modules.
    Revision, offline, cache and dtype options pass through to the loader.
    """
    from transformers import AutoConfig

    config_keys = {
        "revision",
        "cache_dir",
        "local_files_only",
        "token",
        "force_download",
        "subfolder",
        "trust_remote_code",
    }
    config = AutoConfig.from_pretrained(
        model_id, **{key: value for key, value in kwargs.items() if key in config_keys}
    )
    key = (config.model_type, component)
    if key not in LOADERS:
        raise ValueError(f"Unsupported backbone component: {key}")
    module_name, function_name = LOADERS[key]
    module = import_module(module_name)
    loader = getattr(module, function_name)
    return loader(model_id, **kwargs)
