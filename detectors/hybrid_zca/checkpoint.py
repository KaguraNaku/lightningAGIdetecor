"""Strict full-native Hybrid ZCA conversion, retaining every vision/statistic tensor."""

import torch


def translate_native(model, weights):
    expected = model.state_dict()
    result, ignored = {}, []
    wrapped = hasattr(model.backbone, "vision_model")
    for key, value in weights.items():
        if key.startswith("backbone.vision_model."):
            suffix = key.removeprefix("backbone.vision_model.")
            target = ("backbone.vision_model." if wrapped else "backbone.") + suffix
        elif key in model.statistics.state_dict():
            target = "statistics." + key
        elif key.startswith("backbone.text_model.") or key in (
            "backbone.text_projection.weight",
            "backbone.visual_projection.weight",
            "backbone.logit_scale",
        ):
            ignored.append(key)
            continue
        else:
            raise ValueError(f"Unknown native Hybrid ZCA checkpoint key: {key}")
        if target in result or target not in expected:
            raise ValueError(f"Unexpected/duplicate Hybrid ZCA target: {target}")
        if (
            value.shape != expected[target].shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid native Hybrid ZCA tensor: {key}")
        result[target] = value
    if set(result) != set(expected):
        raise ValueError(
            f"Missing native Hybrid ZCA tensors: {sorted(set(expected) - set(result))}"
        )
    return result, {
        "translated_vision_and_statistics": len(result),
        "discarded_unused_full_CLIP_keys": ignored,
        "discard_policy": "Only text model, text/visual projections and logit_scale; all vision and fitted statistics retained",
    }
