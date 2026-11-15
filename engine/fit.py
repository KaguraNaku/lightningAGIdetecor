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


class CheckpointReference:
    """Stable identity for one native file or a fixed-eight NPZ head pair."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)

    def state_dict(self):
        from data.manifest import sha256
        if self.path.is_file():
            return {'path':str(self.path),'sha256':sha256(self.path)}
        import hashlib
        import json
        components = {name:sha256(self.path/name)
                      for name in ('coefficients.npz','mapping.npz')}
        identity = hashlib.sha256(json.dumps(components,sort_keys=True).encode()).hexdigest()
        return {'path':str(self.path),'sha256':identity,'kind':'fixed8_npz_head',
                'components':components}


def restore(path):
    path = Path(path)
    if path.is_dir():
        from detectors.lightning_detetcor.fixed8_task import Fixed8LightningTask
        settings = {"mapping_path": str(path / "mapping.npz"),
                    "coefficients_path": str(path / "coefficients.npz")}
        task = Fixed8LightningTask(settings)
        checkpoint = {"version": 1,
                      "model": {"factory": "detectors.lightning_detetcor.fixed8_task:create_task", **settings},
                      "score": {"direction": "higher_is_fake", "threshold": 0.0},
                      "provenance": {"operation": "read_existing_fixed8_npz"}}
        return task, checkpoint
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path)
    )
    if (
        checkpoint.get("version") != 1
        or checkpoint.get("score", {}).get("direction") != "higher_is_fake"
    ):
        raise ValueError("Unsupported checkpoint version or score direction")
    return create_task(checkpoint["model"], state=checkpoint["task"]), checkpoint
