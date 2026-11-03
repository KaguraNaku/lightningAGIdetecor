"""Strict legacy VideoMAE schemas; plain sinusoid is reconstructed separately."""

import json
from importlib.resources import files
from pathlib import Path

import torch

from backbones.models.videomae_native import position_table
from data.manifest import sha256
from detectors._shared.checkpoint import load_strict, read_weights


SCHEMA = json.loads(files("configs").joinpath("migrations/videomae_pretraining_schema.json").read_text())
ENCODER_SHAPES = {name.removeprefix("videomae."): tuple(row["shape"])
                  for name, row in SCHEMA["tensors"].items() if name.startswith("videomae.")}
DETECTOR_SHAPES = {**{"backbone." + name: shape for name, shape in ENCODER_SHAPES.items()},
                   "fc_norm.weight": (768,), "fc_norm.bias": (768,),
                   "head.weight": (2, 768), "head.bias": (2,)}


def read_native_weights(path):
    path = Path(path).expanduser().resolve()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        weights = load_file(str(path), device="cpu")
        return weights, {"checkpoint": str(path), "checkpoint_sha256": sha256(path),
                         "format": "safetensors"}
    weights, origin = read_weights(path)
    return weights, {**origin, "format": "PyTorch_named_tensor_state"}


def validated(weights, shapes, *, context, convert=True):
    if set(weights) != set(shapes):
        missing = sorted(set(shapes) - set(weights))[:5]
        extra = sorted(set(weights) - set(shapes))[:5]
        raise ValueError(f"{context} strict legacy keys differ; missing={missing}, extra={extra}")
    for name, shape in shapes.items():
        value = weights[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f"{context} tensor shape/type differs: {name}")
        if value.is_meta:
            raise ValueError(f"{context} has unmaterialized weights: {name}")
    return {name: value.float() for name, value in weights.items()} if convert else weights


def initialization_weights(path):
    supplied, origin = read_native_weights(path)
    if set(supplied) == set(ENCODER_SHAPES):
        encoder, discarded, layout = supplied, [], "native_encoder_only"
    elif set(supplied) == set(SCHEMA["tensors"]):
        validated(supplied, {n: tuple(v["shape"]) for n, v in SCHEMA["tensors"].items()}, context="VideoMAE pretraining", convert=False)
        encoder = {name.removeprefix("videomae."): value for name, value in supplied.items() if name.startswith("videomae.")}
        discarded = sorted(name for name in supplied if not name.startswith("videomae."))
        layout = "pinned_published_pretraining_schema"
    else:
        raise ValueError("VideoMAE init_weights needs complete native encoder or pinned pretraining layout; trained detectors use initial_checkpoint")
    encoder = validated(encoder, ENCODER_SHAPES, context="VideoMAE initialization")
    return encoder, {**origin, "layout": layout, "encoder_matched_keys": len(encoder),
                     "discarded_pretraining_keys": discarded,
                     "floating_import_dtype": "float32_as_native_default_constructor",
                     "schema_header_sha256": SCHEMA["header_sha256"],
                     "schema_is_not_tensor_value_verification": True}


def trained_weights(path):
    supplied, origin = read_native_weights(path)
    weights = validated(supplied, DETECTOR_SHAPES, context="VideoMAE trained detector")
    return weights, {**origin, "layout": "native_full_detector",
                     "matched_keys": len(weights), "floating_import_dtype": "float32_as_native_constructor"}


def restore_positions(backbone):
    # A plain attribute is not checked/materialized by nn.load_state_dict.
    with torch.device("cpu"):
        backbone.embeddings.position_embeddings = position_table()
    return backbone


def restore_detector(model, weights):
    # Canonical state is stricter than external float imports: no dtype repair.
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 for v in weights.values()):
        raise ValueError("Saved VideoMAE task weights must remain float32")
    validated(weights, DETECTOR_SHAPES, context="VideoMAE saved detector")
    load_strict(model, weights, assign=True)
    restore_positions(model.backbone)
    return model
