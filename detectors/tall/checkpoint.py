"""Strict complete TALL import and self-contained canonical restoration."""

from functools import lru_cache

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors.tall.detector import TALLDetector


def require_profile():
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("The selected native TALL profile requires default float32")


@lru_cache(maxsize=1)
def detector_schema():
    require_profile()
    with torch.device("meta"):
        model = TALLDetector(
            load_backbone("tall_swin_base", "vision", model_type="tall_swin_base")
        )
    expected = {
        key: (tuple(value.shape), value.dtype)
        for key, value in model.state_dict().items()
    }
    assert len(expected) == 356 and len(list(model.parameters())) == 330
    return expected


def validate_detector_state(weights):
    require_profile()
    expected = detector_schema()
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError(
            "TALL requires all356 exact Detector keys,including position/mask buffers"
        )
    for key, (shape, dtype) in expected.items():
        value = weights[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.is_meta
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or value.dtype != dtype
        ):
            raise ValueError(f"Invalid TALL state shape,dtype or layout: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite TALL state: {key}")
    return weights


def trained_weights(path):
    weights, receipt = read_weights(path)
    validate_detector_state(weights)
    return weights, {
        **receipt,
        "layout": "complete_native_TALL_Detector356",
        "operation": "strict_full_detector_import;does_not_assert_weights_were_trained",
        "dtype_policy": "Exact native dtypes,no cast or missing-buffer reconstruction",
        "pretraining_translation": False,
    }


def restore_detector(weights):
    weights = validate_detector_state(weights)
    with torch.device("meta"):
        model = TALLDetector(
            load_backbone("tall_swin_base", "vision", model_type="tall_swin_base")
        )
    load_strict(model, weights, assign=True)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError(
            "Native TALL restoration must preserve all330 trainable parameters"
        )
    return model
