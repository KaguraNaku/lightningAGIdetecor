"""Strict complete ProDet transport with an explicitly declared native recipe."""

from functools import lru_cache

import torch

from detectors._shared.checkpoint import read_weights, restore_model
from .backbone import EfficientNetB4
from .detector import ProDetDetector


def selected_recipe():
    # These nonpersistent switches cannot be inferred from a state dictionary.
    return {
        "backbone_id": "efficientnetb4",
        "mode": "original",
        "in_channels": 3,
        "num_classes": 2,
        "dropout": 0.0,
        "linear_init": False,
        "cat_manner": "linear",
        "feature_drift": True,
        "feature_operation": True,
        "op_start": 0,
    }


def require_recipe(recipe):
    expected = selected_recipe()
    if (
        not isinstance(recipe, dict)
        or set(recipe) != set(expected)
        or any(
            type(recipe[key]) is not type(value) or recipe[key] != value
            for key, value in expected.items()
        )
    ):
        raise ValueError(
            "ProDet checkpoint requires the explicit selected native linear/drift/mix recipe"
        )
    if torch.get_default_dtype() != torch.float32:
        raise ValueError(
            "The selected native ProDet checkpoint profile requires default float32"
        )


def _detector():
    return ProDetDetector(
        EfficientNetB4(mode="original"),
        feature_drift=True,
        feature_operation=True,
        op_start=0,
    )


@lru_cache(maxsize=1)
def _schema():
    with torch.device("meta"):
        model = _detector()
    assert len(model.state_dict()) == 782 and len(list(model.parameters())) == 488
    return {
        key: (tuple(value.shape), value.dtype, tuple(value.stride()))
        for key, value in model.state_dict().items()
    }


def detector_schema(*, recipe):
    # Check before the cache so a later dtype/recipe change cannot bypass it.
    require_recipe(recipe)
    return dict(_schema())


def validate_detector_state(weights, *, recipe):
    expected = detector_schema(recipe=recipe)
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError(
            "ProDet requires all782 exact Detector keys,including unused backbone head and all BN buffers"
        )
    for key, (shape, dtype, stride) in expected.items():
        value = weights[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or tuple(value.stride()) != stride
        ):
            raise ValueError(
                f"Invalid ProDet CPU state shape,dtype or native stride: {key}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite ProDet state: {key}")
    return weights


def detector_weights(path, *, recipe):
    require_recipe(recipe)
    weights, receipt = read_weights(path)
    validate_detector_state(weights, recipe=recipe)
    return weights, {
        **receipt,
        "recipe": selected_recipe(),
        "layout": "complete_native_ProDet_Detector782",
        "operation": "strict_full_detector_weights_import;does_not_assert_weights_were_trained",
        "dtype_policy": "Exact native CPU dtypes/strides,no casts or missing-buffer reconstruction",
        "backbone_pretraining_translation": False,
        "optimizer_scheduler_or_epoch_resumed": False,
    }


def restore_detector(weights, *, recipe):
    validate_detector_state(weights, recipe=recipe)
    # Native load_state_dict copies into independent constructor allocations.
    # Clone each validated tensor so caller aliases cannot tie unrelated native
    # parameters or mutate the restored model through the input mapping.
    with torch.no_grad():
        independent = {
            key: value.detach().clone(memory_format=torch.contiguous_format)
            for key, value in weights.items()
        }
    model = restore_model(_detector, independent)
    if len(list(model.parameters())) != 488 or any(
        not p.requires_grad for p in model.parameters()
    ):
        raise ValueError(
            "Native ProDet restoration must preserve all488 trainable parameters"
        )
    return model
