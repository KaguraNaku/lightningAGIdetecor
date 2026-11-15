"""Model-independent fitting orchestration and checkpoint envelope."""

from pathlib import Path
import zipfile
import time

import torch

from .runtime import atomic_save, create_task


def fit(task, loader, *, validation_loader=None, runtime, config, output_dir, provenance):
    fit_started = time.perf_counter()
    options = {}
    # Preserve adapters predating validation support when no validation is requested.
    if validation_loader is not None:
        options["validation_loader"] = validation_loader
    report = task.fit(
        loader, runtime=runtime, settings=config["fit"], output_dir=output_dir, **options
    )
    export_started = time.perf_counter()
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
    from engine.training_timing import update_export_timing
    update_export_timing(output_dir, fit_wall=time.perf_counter() - fit_started,
        export_wall=time.perf_counter() - export_started, model_path=path)
    return path, report


def restore(path):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path)
    )
    if (
        checkpoint.get("version") != 1
        or checkpoint.get("score", {}).get("direction") != "higher_is_fake"
    ):
        raise ValueError("Unsupported checkpoint version or score direction")
    return create_task(checkpoint["model"], state=checkpoint["task"]), checkpoint
