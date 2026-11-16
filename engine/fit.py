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


class CheckpointImporter:
    """Import complete native weights without opening datasets or fitting a model."""

    INITIALIZATION_FIELDS = frozenset({
        "initial_checkpoint", "init_weights", "initialization", "base_clip_checkpoint",
    })

    def __init__(self, config_path, *, model_name=None, env_file=None, legacy_weight_mode=None):
        from .config import read_experiment
        from .models import ModelRegistry
        from .resources import ResourceResolver

        path = Path(config_path).expanduser().resolve(strict=True)
        config = read_experiment(path, require_fit=False, env_file=env_file)
        model = dict(config["model"])
        if model_name is not None:
            model = ModelRegistry.select(model_name, model)
        ModelRegistry.validate_factory(model)
        native = {ModelRegistry.get(name).factory for name in ModelRegistry.NATIVE_CLASSES}
        if ModelRegistry.resolve_factory(model["factory"]) not in native:
            raise ValueError("Raw checkpoint import requires a native detector Task")
        # Full detector weights replace initialization assets, not the declared
        # architecture or preprocessing. Do not require unused ImageNet files.
        model = {key: value for key, value in model.items()
                 if key not in self.INITIALIZATION_FIELDS}
        if legacy_weight_mode is not None:
            if ModelRegistry.method_for(model) != "effort":
                raise ValueError("--legacy-weight-mode applies only to Effort")
            model["legacy_weight_mode"] = legacy_weight_mode
        self.settings = ResourceResolver(path.parent, env_file=env_file).model_settings(model)
        self.threshold = config["evaluation"]["threshold"]

    def convert(self, source, destination):
        from detectors._shared.checkpoint import convert_checkpoint

        return convert_checkpoint(self.settings, source, destination, threshold=self.threshold)


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
        from detectors.lightning.task import LightningTask
        settings = {"mapping_path": str(path / "mapping.npz"),
                    "coefficients_path": str(path / "coefficients.npz")}
        task = LightningTask(settings)
        checkpoint = {"version": 1,
                      "model": {"factory": "detectors.lightning.task:LightningTask", **settings},
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
