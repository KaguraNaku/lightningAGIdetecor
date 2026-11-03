"""STIL native two-stage initialization,distinct from strict canonical restore."""

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import zipfile

import torch

from backbones.load_backbone import load_backbone
from data.manifest import sha256
from detectors._shared.checkpoint import load_strict
from .detector import STILDetector

SCHEMA = json.loads(
    files("configs").joinpath("migrations/stil_scnet_pretraining.json").read_text()
)


def _profile():
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Native STIL profile requires default float32")


def read_native_state(path):
    """Keep native raw keys and OrderedDict._metadata;no wrapper/alias repair."""
    path = Path(path).expanduser().resolve()
    weights = torch.load(
        path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path)
    )
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor)
            for k, v in weights.items()
        )
    ):
        raise ValueError(
            "STIL needs a raw nonempty named tensor state,without an envelope"
        )
    for key, value in weights.items():
        if value.is_meta or value.layout != torch.strided:
            raise ValueError(f"STIL needs materialized strided tensors: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite STIL checkpoint tensor: {key}")
    return weights, {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256(path),
        "raw_keys_preserved": True,
        "OrderedDict_metadata_preserved": hasattr(weights, "_metadata"),
    }


def _key_hash(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def initialize_backbone(backbone, path):
    """Copy the pinned SCNet layout into an already randomly initialized model."""
    _profile()
    weights, origin = read_native_state(path)
    if set(weights) != set(SCHEMA["tensors"]):
        raise ValueError("STIL pretraining requires the complete raw SCNet1000 layout")
    for key, value in weights.items():
        expected = SCHEMA["tensors"][key]
        if (
            list(value.shape) != expected["shape"]
            or str(value.dtype) != expected["dtype"]
        ):
            raise ValueError(f"STIL pretraining shape/dtype differs: {key}")
    copied = backbone.load_state_dict(weights, strict=False)
    expected = SCHEMA["native_load_result"]
    if (
        _key_hash(copied.missing_keys) != expected["missing_sha256"]
        or _key_hash(copied.unexpected_keys) != expected["unexpected_sha256"]
    ):
        raise ValueError("STIL native nonstrict pretraining result differs")
    return {
        **origin,
        "layout": "native_SCNet1000_public_schema",
        "published_asset_sha256_matches": origin["checkpoint_sha256"]
        == SCHEMA["source_sha256"],
        "matched_keys": expected["matched"],
        "missing_keys": copied.missing_keys,
        "unexpected_keys": copied.unexpected_keys,
        "load_policy": "Original nonstrict copy,no SCConv k2 remapping;missing temporal/spatial modules retain constructor draws",
    }


def apply_native_override(model, path):
    """Original optional pretrained-path translation after binary head creation."""
    weights, origin = read_native_state(path)
    mapped = {}
    for key, value in weights.items():
        target = "base_model." + key.replace("base_", "").replace("model.", "")
        if target in mapped:
            raise ValueError(f"Ambiguous STIL native key replacement: {target}")
        mapped[target] = value
    expected = model.state_dict()
    overlap = set(mapped) & set(expected)
    if not overlap:
        raise ValueError("STIL native override has no matching parameters")
    for key in overlap:
        if mapped[key].shape != expected[key].shape:
            raise ValueError(f"STIL native override shape differs: {key}")
    copied = model.load_state_dict(mapped, strict=False)
    return {
        **origin,
        "layout": "native_optional_pretrained_path",
        "key_rule": "literal base_ replacement,then literal model. replacement,then base_model. prefix",
        "matched_keys": len(overlap),
        "missing_keys": copied.missing_keys,
        "unexpected_keys": copied.unexpected_keys,
        "metadata_policy": "Native dictionary comprehensions discard OrderedDict metadata",
        "input_guards": "Reject collisions,zero overlap and mismatched shapes before copy",
    }


def initialize_detector(pretraining_path, *, native_override=None):
    """Mandatory SCNet copy -> fresh binary head -> optional native override."""
    _profile()
    if pretraining_path is None:
        raise ValueError("STIL requires explicit local SCNet pretraining")
    backbone = load_backbone("stil_scnet50", "vision", model_type="stil_scnet50")
    pretraining = initialize_backbone(backbone, pretraining_path)
    model = STILDetector(backbone)
    override = (
        apply_native_override(model.model, native_override)
        if native_override is not None
        else None
    )
    return model, {
        "pretraining": pretraining,
        "native_override": override,
        "operation": "initialization,not training",
        "implicit_download": False,
        "load_order": "Random SCNet1000,public layout copy,fresh Linear2048to2,optional literal-key override",
    }


@lru_cache(maxsize=1)
def _detector_schema():
    _profile()
    with torch.device("meta"):
        model = STILDetector(
            load_backbone("stil_scnet50", "vision", model_type="stil_scnet50")
        )
    return {
        name: (tuple(value.shape), value.dtype)
        for name, value in model.state_dict().items()
    }


def validate_detector_state(weights, *, canonical):
    _profile()
    expected = _detector_schema()
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError(
            "STIL requires all1548 exact Detector keys,including BN counters"
        )
    supplied = {}
    for key, (shape, dtype) in expected.items():
        value = weights[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.is_meta
            or value.layout != torch.strided
            or tuple(value.shape) != shape
        ):
            raise ValueError(f"Invalid STIL Detector tensor: {key}")
        if canonical:
            if value.dtype != dtype:
                raise ValueError(f"Canonical STIL dtype must remain {dtype}: {key}")
        elif dtype == torch.float32:
            if not value.is_floating_point():
                raise ValueError(f"STIL floating parameter required: {key}")
            value = value.float()
        elif value.dtype != dtype:
            raise ValueError(f"STIL BN counter dtype must remain {dtype}: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite STIL Detector value after import: {key}")
        supplied[key] = value
    return supplied


def trained_weights(path):
    weights, origin = read_native_state(path)
    weights = validate_detector_state(weights, canonical=False)
    return weights, {
        **origin,
        "layout": "native_full_STIL_Detector",
        "matched_keys": len(weights),
        "floating_import_dtype": "float32_as_native_constructor",
    }


def restore_detector(weights, *, canonical=True):
    """Full-state restore without pretraining IO or consumed CPU random draws."""
    weights = validate_detector_state(weights, canonical=canonical)
    with torch.device("meta"):
        model = STILDetector(
            load_backbone("stil_scnet50", "vision", model_type="stil_scnet50")
        )
    return load_strict(model, weights, assign=True)
