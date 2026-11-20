"""Strict state loading tools; method-specific translations live with each model."""

from pathlib import Path
import math
import zipfile

import torch

from data.manifest import sha256


def read_weights(path, *, encoding=None):
    path = Path(path).expanduser().resolve()
    pickle_options = {"encoding": encoding} if encoding is not None else {}
    envelope = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=zipfile.is_zipfile(path),
        **pickle_options,
    )
    if not isinstance(envelope, dict):
        raise ValueError("Checkpoint must be a named state dictionary or envelope")
    weights = envelope.get("state_dict", envelope)
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor)
            for k, v in weights.items()
        )
    ):
        raise ValueError("Checkpoint weights must contain named tensors")
    if all(k.startswith("module.") for k in weights):
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
    return weights, {"checkpoint": str(path), "checkpoint_sha256": sha256(path)}


def load_strict(model, weights, *, assign=False):
    model.load_state_dict(weights, strict=True, assign=assign)
    for module in model.modules():
        # CLIP's position indices are nonpersistent and are not saved in state_dict.
        if hasattr(module, "position_ids") and hasattr(module, "position_embedding"):
            module.position_ids = torch.arange(
                module.position_embedding.num_embeddings, device="cpu"
            ).expand((1, -1))
    missing = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_meta
    ]
    if missing:
        raise ValueError(f"Unmaterialized model state: {missing}")
    return model


def restore_model(build, weights):
    """Construct the model on meta and materialize it from its saved tensors."""
    with torch.device("meta"):
        model = build()
    return load_strict(model, weights, assign=True)


def convert_checkpoint(settings, source, destination, *, threshold=0.5):
    """Convert via the selected task's strict importer into the engine envelope."""
    from engine.runtime import atomic_save, create_task

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source == destination or destination.exists():
        raise ValueError("Conversion destination must be a new file")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Probability threshold must be finite and in [0, 1]")
    model = {**settings, "initial_checkpoint": str(source)}
    task = create_task(model)
    state = task.state_dict()
    model.pop("initial_checkpoint")
    atomic_save(
        destination,
        {
            "version": 1,
            "model": model,
            "task": state,
            "score": {"direction": "higher_is_fake", "threshold": threshold},
            "provenance": state["origin"],
            "fit_report": {"operation": "legacy_import"},
        },
        torch_format=True,
    )
    return destination


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Convert image detector weights for engine.py evaluate"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-weight-mode", choices=["merged", "unmerged"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = dict(config["model"])
    if args.legacy_weight_mode is not None:
        settings["legacy_weight_mode"] = args.legacy_weight_mode
    print(
        convert_checkpoint(
            settings,
            args.input,
            args.output,
            threshold=config.get("evaluation", {}).get("threshold", 0.5),
        )
    )


if __name__ == "__main__":
    main()
