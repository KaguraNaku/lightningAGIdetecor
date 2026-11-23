"""Lazy supervised ViT encoder loading, without a randomly initialized pooler."""


def load_vit_vision(model_id, **kwargs):
    from transformers import ViTModel

    if kwargs.pop("add_pooling_layer", False):
        raise ValueError("ViT features use the native final CLS, not a learned pooler")
    model, info = ViTModel.from_pretrained(
        model_id, add_pooling_layer=False, output_loading_info=True, **kwargs
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError("ViT encoder checkpoint is incomplete: " + repr(info))
    unexpected = info.get("unexpected_keys", [])
    if any(not key.startswith("classifier.") for key in unexpected):
        raise ValueError("Unexpected ViT encoder tensors: " + repr(unexpected))
    return model
