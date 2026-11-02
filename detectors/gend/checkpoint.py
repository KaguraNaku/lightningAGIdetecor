"""Strict raw-GenD or DeepfakeBench wrapper imports, without prefix guessing."""

from pathlib import Path

from safetensors import safe_open

from data.manifest import sha256
from detectors._shared.checkpoint import read_weights


def read_gend_weights(path):
    path = Path(path).expanduser()
    is_safetensors = path.suffix == ".safetensors"
    path = path.resolve()
    if is_safetensors:
        with safe_open(path, framework="pt", device="cpu") as stream:
            weights = {name: stream.get_tensor(name) for name in stream.keys()}
        receipt = {"checkpoint": str(path), "checkpoint_sha256": sha256(path)}
    else:
        weights, receipt = read_weights(path)
    if all(name.startswith("module.") for name in weights):
        weights = {
            name.removeprefix("module."): value for name, value in weights.items()
        }
    if any(name.startswith("feature_extractor.") for name in weights):
        weights = {f"model.{name}": value for name, value in weights.items()}
        receipt["checkpoint_layout"] = "raw_gend"
    else:
        receipt["checkpoint_layout"] = "deepfakebench_wrapper"
    return weights, receipt
