"""Model-independent fitting orchestration and checkpoint envelope."""

from copy import deepcopy
from pathlib import Path
import zipfile
import time

import torch

from .models import ModelRegistry
from .runtime import atomic_save, create_task, unpack_task_state


def _read_training_parent(reference):
    from data.manifest import sha256

    path = Path(reference["path"])
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Training parent checkpoint checksum differs: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _initial_ufd_detector(model, policy, method):
    adaptation = {
        "regime": policy["regime"],
        "source": "UFD/ProGAN",
        "optimizer": "Fresh native optimizer per fit",
    }
    if method == "ucf":
        from detectors.ucf.task import UFD_LABEL_MAP, expand_progan_head

        model = {**model, "specific_classes": 6, "specific_label_map": dict(UFD_LABEL_MAP)}
        adaptation["UCF_domain_map"] = dict(UFD_LABEL_MAP)
    if method == "forada":
        adaptation["ForAda_supervision"] = "whole-image mask for fully generated images;native losses retained"

    parent = policy.get("parent_checkpoint")
    if parent is None:
        return create_task(model), model, adaptation

    reference = {"path": parent, "sha256": policy["parent_sha256"]}
    saved = _read_training_parent(reference)
    if ModelRegistry.resolve_factory(saved["model"]["factory"]) != model["factory"]:
        raise ValueError("FF++ parent method differs from the selected detector")
    task = create_task(saved["model"], state=saved["task"])
    validate_parent = getattr(task, "validate_training_parent", None)
    if callable(validate_parent):
        validate_parent(model)
    if method == "ucf":
        expand_progan_head(task)
    adaptation["parent"] = {**reference, "selected_epoch": saved["fit_report"]["best_epoch"]}
    return task, model, adaptation


def _continue_ufd_detector(reference, expected_factory):
    saved = _read_training_parent(reference)
    declaration, saved_state = saved["model"], saved["task"]
    legacy = ModelRegistry.LEGACY_UFD_FACTORIES.get(declaration["factory"])
    if legacy == "continual":
        # A historical first-stage import changed the factory but retained its
        # original FF++ to UFD adaptation receipt.
        adaptation = saved_state["adaptation"]
    elif legacy is None and saved.get("fit_report", {}).get("continuation"):
        adaptation = saved_state.get("origin", {}).get("UFD_adaptation", {})
        if adaptation.get("regime") != "sequential_no_replay":
            raise ValueError("Previous selected model lacks its UFD continuation provenance")
    else:
        raise ValueError("A later stage requires the previous selected continual model")
    model, state = unpack_task_state(declaration, saved_state)
    if ModelRegistry.resolve_factory(model["factory"]) != expected_factory:
        raise ValueError("Previous stage model belongs to a different detector")
    return create_task(model, state=state), model, deepcopy(adaptation)


def create_training_task(config):
    """Use the native task for every fit and apply UFD settings at the engine boundary."""
    model, settings = config["model"], config["fit"]
    legacy = ModelRegistry.LEGACY_UFD_FACTORIES.get(model["factory"])
    policy = settings.get("ufd")
    if legacy is None and policy is None:
        return create_task(model)
    if legacy is not None:
        if policy is not None:
            raise ValueError("Choose either fit.ufd or a legacy UFD model declaration")
        ModelRegistry.validate_factory(model)
        policy = {key: model[key] for key in (
            "regime", "parent_checkpoint", "parent_sha256", "previous_stage_model",
        ) if key in model}
        model = model.get("base_model") or {"factory": ModelRegistry.get(model["method"]).factory}
    elif config["inputs"][settings["input"]]["dataset"] != "UniversalFakeDetect":
        raise ValueError("fit.ufd requires a UniversalFakeDetect training input")

    method = ModelRegistry.method_for(model)
    if method not in ModelRegistry.NATIVE_CLASSES:
        raise ValueError("UFD fitting requires a native detector")
    factory = ModelRegistry.get(method).factory
    if ModelRegistry.resolve_factory(model["factory"]) != factory:
        raise ValueError("UFD fitting requires the selected detector's native factory")
    model = {**model, "factory": factory}
    previous = policy.get("previous_stage_model")
    parent = policy.get("parent_checkpoint")
    stage = settings.get("continuation")
    if legacy == "continual" or (legacy is None and stage is not None):
        if not stage or ((stage["stage_index"] > 0) != bool(previous)):
            raise ValueError("Continual fitting requires aligned explicit stage and previous-model references")
        if bool(previous) == bool(parent):
            raise ValueError("Stage zero uses an FF++ parent; later stages use the previous UFD model")
    elif previous:
        raise ValueError("A previous UFD model requires fit.continuation")
    elif not legacy and policy.get("regime") != ("FFpp_to_UFD" if parent else "fresh_UFD"):
        raise ValueError("UFD regime must match the declared checkpoint initialization")

    if previous:
        task, model, adaptation = _continue_ufd_detector(previous, factory)
    else:
        task, model, adaptation = _initial_ufd_detector(model, policy, method)
    if stage is not None:
        adaptation.update(
            regime="sequential_no_replay",
            optimizer="Native optimizer initialized once at FF++→UFD;all state restored from each selected prior epoch",
            worker_rng="Nonpersistent workers restart at epoch boundaries from carried loader generator",
        )
    task.origin["UFD_adaptation"] = adaptation
    config["model"] = {**model, "factory": factory}
    settings["ufd"] = policy
    return task


def fit(task, loader, *, validation_loader=None, runtime, config, output_dir, provenance):
    fit_started = time.perf_counter()
    options = {}
    # Preserve adapters predating validation support when no validation is requested.
    if validation_loader is not None:
        options["validation_loader"] = validation_loader
    report = task.fit(
        loader, runtime=runtime, settings=config["fit"], output_dir=output_dir, **options
    )
    if config["fit"].get("ufd") is not None:
        report = {**report, "adaptation": deepcopy(task.origin["UFD_adaptation"])}
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
        if ModelRegistry.method_for(model) == "prodet":
            # Training-only pair bindings do not participate in inference or
            # complete-weight import. Preserve all architecture/score settings.
            model.pop("training_neighbors", None)
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
    """Stable identity for one engine checkpoint file."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("Expected an engine checkpoint file")

    def state_dict(self):
        from data.manifest import sha256
        return {"path": str(self.path), "sha256": sha256(self.path)}


def restore(path):
    path = Path(path).expanduser()
    if path.is_dir():
        raise ValueError("Expected an engine checkpoint file")
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path)
    )
    if (
        checkpoint.get("version") != 1
        or checkpoint.get("score", {}).get("direction") != "higher_is_fake"
    ):
        raise ValueError("Unsupported checkpoint version or score direction")
    return create_task(checkpoint["model"], state=checkpoint["task"]), checkpoint
