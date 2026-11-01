from importlib import import_module
from transformers import AutoConfig
from .registry import LOADERS

def load_backbone(model_id:str, component:str="", *, model_type=None, **kwargs):
    if model_type is None:
        config = AutoConfig.from_pretrained(
            model_id, **{k: v for k, v in kwargs.items() if k in ("revision", "local_files_only")}
        )
        model_type = config.model_type
    entry = LOADERS[(model_type, component)]
    if callable(entry):
        loader = entry
    else:
        module_name, function_name = entry
        module = import_module(module_name)
        loader = getattr(module, function_name)
    return loader(model_id, **kwargs)

