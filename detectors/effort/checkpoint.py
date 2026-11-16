"""Strict legacy import and conversion into the shared engine checkpoint format.

Conversion is an offline artifact utility, not an additional experiment runner.
Training checkpoints are retained for diagnostics; exact training resume is not
implemented (in particular, prefetched worker RNG states are not restored).
"""

import argparse
from pathlib import Path

import torch

from data.manifest import sha256
from engine.runtime import atomic_save


def read_legacy(path):
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Effort checkpoint must be a state dictionary or training envelope")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Effort legacy state_dict must contain named tensors")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    receipt = {"checkpoint": str(path), "checkpoint_sha256": sha256(path)}
    for name in ("epoch", "protocol", "record"):
        if name in checkpoint:
            receipt[name] = checkpoint[name]
    return state, receipt


def convert_checkpoint(source, destination, *, backbone_id, revision=None, local_files_only=False):
    from .task import EffortTask

    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if source == destination or destination.exists():
        raise ValueError("Conversion destination must be a new file")
    settings = {
        "backbone_id": str(backbone_id), "initial_checkpoint": str(source),
        "local_files_only": local_files_only,
    }
    if revision is not None:
        settings["revision"] = revision
    task = EffortTask(settings)
    # Omit the source path from construction settings: restoration is self-contained.
    model = {k: v for k, v in settings.items() if k != "initial_checkpoint"}
    model["factory"] = "detectors.effort.task:EffortTask"
    envelope = {
        "version": 1, "model": model, "task": task.state_dict(),
        "score": {"direction": "higher_is_fake", "threshold": 0.5},
        "provenance": task.origin, "fit_report": {"operation": "legacy_import"},
    }
    atomic_save(destination, envelope, torch_format=True)
    return destination


def main():
    from .task import BACKBONE_ID

    parser = argparse.ArgumentParser(description="Convert author Effort weights for engine.py evaluate")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-id", default=BACKBONE_ID)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(convert_checkpoint(
        args.input, args.output, backbone_id=args.backbone_id,
        revision=args.revision, local_files_only=args.local_files_only,
    ))


if __name__ == "__main__":
    main()
