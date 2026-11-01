"""Model-independent fitting orchestration and checkpoint envelope."""

from pathlib import Path

import torch

from .runtime import atomic_save, create_task


def fit(task, loader, *, validation_loader=None, runtime, config, output_dir, provenance):
    options = {}
    # Preserve adapters predating validation support when no validation is requested.
    if validation_loader is not None:
        options["validation_loader"] = validation_loader
    report = task.fit(
        loader, runtime=runtime, settings=config["fit"], output_dir=output_dir, **options
    )
    checkpoint = {
        "version": 1,
        "model": config["model"],
        "task": task.state_dict(),
        "score": {
            "direction": "higher_is_fake",
            "threshold": float(config["evaluation"]["threshold"]),
        },
        "provenance": provenance,
        "fit_report": report,
    }
    path = Path(output_dir) / "model.pth"
    atomic_save(path, checkpoint, torch_format=True)
    return path, report


def restore(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("version") != 1
        or checkpoint.get("score", {}).get("direction") != "higher_is_fake"
    ):
        raise ValueError("Unsupported checkpoint version or score direction")
    return create_task(checkpoint["model"], state=checkpoint["task"]), checkpoint
