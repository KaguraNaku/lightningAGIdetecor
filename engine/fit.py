"""Model-independent fitting orchestration and checkpoint envelope."""

from copy import deepcopy
from pathlib import Path
import zipfile

import torch

from .models import ModelRegistry
from .resources import resolve_asset
from .runtime import atomic_save, create_task, task_factory


def read_checkpoint(reference, *, require_checksum=False, encoding=None):
    """Load saved tensors on CPU, using mmap for zip-format checkpoints."""
    path = resolve_asset(reference, require_checksum=require_checksum)
    if not path.is_file():
        raise ValueError("Expected an engine checkpoint file")
    options = {"encoding": encoding} if encoding is not None else {}
    return torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path), **options)


def _initial_ufd_detector(model, policy):
    adaptation = {
        "regime": policy["regime"],
        "source": "UFD/ProGAN",
        "optimizer": "Fresh native optimizer per fit",
    }
    if (model_settings := getattr(task_factory(model), "ufd_model_settings", None)) is not None:
        model = {**model, **model_settings()}
    parent = policy.get("parent_checkpoint")
    if parent is None:
        task = create_task(model)
    else:
        reference = {"path": parent, "sha256": policy["parent_sha256"]}
        saved = read_checkpoint(reference, require_checksum=True)
        if ModelRegistry.resolve_factory(saved["model"]["factory"]) != model["factory"]:
            raise ValueError("FF++ parent method differs from the selected detector")
        task = create_task(saved["model"], state=saved["task"])
        if (validate_parent := getattr(task, "validate_training_parent", None)) is not None:
            validate_parent(model)
        adaptation["parent"] = {**reference, "selected_epoch": saved["fit_report"]["best_epoch"]}
    if (adapt := getattr(task, "configure_ufd_training", None)) is not None:
        adapt(adaptation, warm_start=parent is not None)
    return task, model, adaptation


def create_training_task(config):
    model, settings = config["model"], config["fit"]
    if "continuation" in settings:
        raise ValueError("Use --checkpoint to resume training")
    if settings.get("checkpoint") is not None:
        return _resume_training_task(config)
    policy = settings.get("ufd")
    if policy is None:
        return create_task(model)
    if set(policy) - {"regime", "parent_checkpoint", "parent_sha256"}:
        raise ValueError("fit.ufd accepts only regime, parent_checkpoint and parent_sha256")
    if config["inputs"][settings["input"]]["dataset"] != "UniversalFakeDetect":
        raise ValueError("fit.ufd requires a UniversalFakeDetect training input")

    method = ModelRegistry.method_for(model)
    if method not in ModelRegistry.NATIVE_CLASSES:
        raise ValueError("UFD fitting requires a native detector")
    factory = ModelRegistry.get(method).factory
    if ModelRegistry.resolve_factory(model["factory"]) != factory:
        raise ValueError("UFD fitting requires the selected detector's native factory")
    parent = policy.get("parent_checkpoint")
    if policy.get("regime") != ("FFpp_to_UFD" if parent else "fresh_UFD"):
        raise ValueError("UFD regime must match the declared checkpoint initialization")

    task, model, adaptation = _initial_ufd_detector({**model, "factory": factory}, policy)
    task.origin["UFD_adaptation"] = adaptation
    config["model"] = model
    return task


def _resume_training_task(config):
    model, settings = config["model"], config["fit"]
    path = Path(settings["checkpoint"])
    selected = ModelRegistry.method_for(model)
    if (resume := getattr(task_factory(model), "resume_model_settings", None)) is not None:
        config["model"] = resume(model, path)
        return create_task(config["model"])
    saved = read_model_checkpoint(path)
    if selected != ModelRegistry.method_for(saved["model"]):
        raise ValueError("Checkpoint belongs to a different detector")
    if "training_state" not in saved:
        raise ValueError("This model.pth does not record a resumable training state")
    source = config["inputs"][settings["input"]]["dataset"]
    prior_sources = {item["dataset"] for item in saved["provenance"].get("inputs", {}).values()
                     if item.get("split") == "train"}
    if prior_sources and prior_sources != {source}:
        raise ValueError("Resume on the same dataset. Use fit.ufd for FF++ to UFD initialization.")
    saved_model = saved["model"]
    task = create_task(saved_model, state=saved["task"])
    if (bind := getattr(task, "bind_resume_input", None)) is not None:
        saved_model = {**saved_model, **bind(model)}
    if "UFD_adaptation" not in task.origin:
        settings.pop("ufd", None)
    config["model"] = saved_model
    settings["resume_state"] = {**saved["training_state"], "path": str(path.parent / "best.pth")}
    return task


def fit(task, loader, *, validation_loader=None, runtime, config, output_dir, provenance):
    options = {}
    # Preserve adapters predating validation support when no validation is requested.
    if validation_loader is not None:
        options["validation_loader"] = validation_loader
    report = task.fit(
        loader, runtime=runtime, settings=config["fit"], output_dir=output_dir, **options
    )
    if config["fit"].get("ufd") is not None:
        report = {**report, "adaptation": deepcopy(task.origin["UFD_adaptation"])}
    checkpoint_id = report.pop("training_checkpoint_id", None)
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
    if checkpoint_id is not None:
        checkpoint["training_state"] = {"id": checkpoint_id, "epoch": report["best_epoch"]}
    path = Path(output_dir) / "model.pth"
    atomic_save(path, checkpoint, torch_format=True)
    return path, report


class CheckpointImporter:
    """Import complete native weights without opening datasets or fitting a model."""

    INITIALIZATION_FIELDS = frozenset({
        "initial_checkpoint", "init_weights", "initialization",
    })

    def __init__(self, config_path, *, model_name=None, env_file=None):
        from .config import read_experiment
        from .models import ModelRegistry
        from .resources import ResourceResolver

        path = Path(config_path).expanduser().resolve(strict=True)
        config = read_experiment(path, require_fit=False, env_file=env_file, model_name=model_name)
        model = dict(config["model"])
        ModelRegistry.validate_factory(model)
        native = {ModelRegistry.get(name).factory for name in ModelRegistry.NATIVE_CLASSES}
        if ModelRegistry.resolve_factory(model["factory"]) not in native:
            raise ValueError("Raw checkpoint import requires a native detector Task")
        # Full detector weights replace initialization assets, not the declared
        # architecture or preprocessing. Do not require unused ImageNet files.
        factory = task_factory(model)
        excluded = self.INITIALIZATION_FIELDS.union(getattr(factory, "import_ignored_settings", ()))
        model = {key: value for key, value in model.items() if key not in excluded}
        self.settings = ResourceResolver(path.parent, env_file=env_file).model_settings(
            model, file_fields=getattr(factory, "model_file_fields", ()),
        )
        self.threshold = config["evaluation"]["threshold"]

    def convert(self, source, destination):
        from detectors._shared.checkpoint import convert_checkpoint

        return convert_checkpoint(self.settings, source, destination, threshold=self.threshold)


class CheckpointReference:
    """Stable identity for one engine checkpoint file."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("Expected an engine checkpoint file")

    def state_dict(self):
        from data.manifest import sha256
        return {"path": str(self.path), "sha256": sha256(self.path)}


def read_model_checkpoint(path):
    checkpoint = read_checkpoint(path)
    if (
        checkpoint.get("version") != 1
        or checkpoint.get("score", {}).get("direction") != "higher_is_fake"
    ):
        raise ValueError("Unsupported checkpoint version or score direction")
    return checkpoint


def restore(path):
    checkpoint = read_model_checkpoint(path)
    return create_task(checkpoint["model"], state=checkpoint["task"]), checkpoint
