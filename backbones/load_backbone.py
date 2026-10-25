from importlib import import_module
from transformers import AutoConfig
from .registry import LOADERS

def load_backbone(model_id:str, component:str=""):
    config = AutoConfig.from_pretrained(model_id)
    module_name, function_name = LOADERS[
        (config.model_type, component)
    ]
    module = import_module(module_name)
    loader = getattr(module, function_name)
    return loader(model_id)

