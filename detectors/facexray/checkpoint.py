"""Literal author HRNet initialization: load trunk, ignore classifier, retain fc."""

from pathlib import Path
import zipfile

import torch

from data.manifest import sha256


def initialize_backbone(model, path):
    path = Path(path).expanduser().resolve()
    weights = torch.load(
        path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path)
    )
    expected = model.state_dict()
    retained = {"fc.weight", "fc.bias"}
    unused = {"classifier.weight", "classifier.bias"}
    if (
        not isinstance(weights, dict)
        or set(weights) != (set(expected) - retained) | unused
    ):
        raise ValueError(
            "FaceXray initialization requires the original unprefixed HRNet ImageNet trunk/classifier layout"
        )
    if any(not isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("FaceXray initialization must contain named tensors only")
    for name in set(expected) & set(weights):
        if expected[name].shape != weights[name].shape:
            raise ValueError(f"FaceXray pretrained shape mismatch: {name}")
    incompatible = model.load_state_dict(weights, strict=False)
    if (
        set(incompatible.missing_keys) != retained
        or set(incompatible.unexpected_keys) != unused
    ):
        raise ValueError("FaceXray native initialization key accounting changed")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "matched_keys": len(set(expected) & set(weights)),
        "retained_fresh_unused_fc": sorted(retained),
        "ignored_pretrained_classifier": sorted(unused),
        "source_behavior": "Literal strictFalse: classifier weights are NOT renamed to fc; decoder and unused fc keep source constructor RNG/state",
    }
