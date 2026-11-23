"""Resolve the shared config into the settings for one engine run."""

from copy import deepcopy
import math
from pathlib import Path
import re

import yaml

from data.config import default_env_file, frames_for

from .models import ModelRegistry
from .resources import ResourceResolver


def _select_experiment(config, model_name, training_input, *, require_fit):
    if model_name is None:
        raise ValueError("Choose --model to use the shared config")
    spec = ModelRegistry.get(model_name)
    profile = config["models"][model_name]
    fit = {**config.get("fit", {}), **profile.get("fit", {})}
    if training_input is not None:
        fit["input"] = training_input
    source = config["inputs"].get(fit.get("input"), {})
    model = {
        "factory": spec.factory,
        **profile.get("settings", {}),
        **source.get("model_settings", {}).get(model_name, {}),
    }
    if (
        require_fit
        and model_name in ModelRegistry.NATIVE_CLASSES
        and source.get("dataset") == "UniversalFakeDetect"
    ):
        fit.setdefault("ufd", {"regime": "fresh_UFD"})

    sampling = config.get("video_sampling", {})
    stage_options = {
        "train": {**config.get("train_loader", {}), **profile.get("train_loader", {})},
        "val": profile.get("validation_loader", {}),
        "test": profile.get("test_loader", {}),
    }
    inputs = {}
    for name, item in config["inputs"].items():
        item = {key: value for key, value in item.items() if key != "model_settings"}
        manifest = item["manifest"]
        if "{frames}" in manifest:
            count = frames_for(item["split"], sampling)
            if (
                config["datasets"][item["dataset"]]["media_type"] != "video"
                or count is None
            ):
                raise ValueError(
                    f"Input {name} requires a video dataset and a configured frame count"
                )
            manifest = manifest.replace("{frames}", str(count))
        inputs[name] = {**stage_options[item["split"]], **item, "manifest": manifest}
    return {
        "version": config["version"],
        "seed": config["seed"],
        "datasets": config["datasets"],
        "video_sampling": sampling,
        "runtime": {**config.get("runtime", {}), **profile.get("runtime", {})},
        "model": model,
        "fit": fit,
        "inputs": inputs,
        "loader": {**config.get("loader", {}), **profile.get("loader", {})},
        "evaluation": {**config.get("evaluation", {}), **profile.get("evaluation", {})},
    }


def read_experiment(
    path,
    *,
    require_fit=True,
    variables=None,
    env_file=None,
    model_name=None,
    training_input=None,
    validation_input=None,
    evaluation_inputs=None,
    checkpoint=None,
    public=False,
):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("Experiment must be a mapping with version: 1")
    if "models" in config:
        config = _select_experiment(
            config, model_name, training_input, require_fit=require_fit
        )
        if env_file is None:
            default_env = default_env_file()
            if default_env.is_file():
                env_file = default_env
    config = deepcopy(config)
    for section in ("model", "loader", "fit", "evaluation", "runtime"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    model = config.get("model", {})
    if model_name is not None:
        config["model"] = model = ModelRegistry.select(model_name, model)
    if public:
        ModelRegistry.validate_factory(model)
    if require_fit and checkpoint is not None:
        config.setdefault("fit", {})["checkpoint"] = str(
            Path(checkpoint).expanduser().resolve()
        )
    factory = model.get("factory", "")
    if require_fit and (
        not isinstance(factory, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", factory)
    ):
        raise ValueError("model.factory must be module:callable")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("inputs must be a nonempty mapping")
    defaults = config.setdefault("loader", {})
    for name, item in inputs.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise ValueError(f"Invalid input name: {name!r}")
        if name in {
            "fit",
            "run.json",
            "run.log",
            "resolved-config.yaml",
            "summary.csv",
            "summary.partial.csv",
            "timing.json",
        }:
            raise ValueError(f"Reserved input name: {name}")
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k), str) and item[k]
            for k in ("dataset", "split", "manifest")
        ):
            raise ValueError(f"Input {name} requires dataset, split and manifest")
        if item["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split for {name}")
        options = {**defaults, **item}
        for key, minimum in (
            ("batch_size", 1),
            ("num_workers", 0),
            ("prefetch_factor", 1),
            ("decode_error_patience", 1),
        ):
            value = options.get(
                key,
                {
                    "batch_size": 32,
                    "num_workers": 0,
                    "prefetch_factor": 2,
                    "decode_error_patience": 10,
                }[key],
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}.{key} must be an integer >= {minimum}")
        for key in ("shuffle", "pin_memory", "persistent_workers"):
            if key in options and type(options[key]) is not bool:
                raise ValueError(f"{name}.{key} must be a boolean")
    if require_fit:
        fit = config.setdefault("fit", {})
        if training_input is not None:
            fit["input"] = training_input
        if validation_input is not None:
            fit["validation_input"] = validation_input
        fit.setdefault("input", "calibration")
        if (
            not isinstance(fit["input"], str)
            or fit["input"] not in inputs
            or inputs[fit["input"]]["split"] != "train"
        ):
            raise ValueError("fit.input must name a train input")
        validation = fit.get("validation_input")
        if validation is not None and (
            not isinstance(validation, str)
            or validation not in inputs
            or inputs[validation]["split"] != "val"
        ):
            raise ValueError("fit.validation_input must name a val input")
        if (
            validation is not None
            and ModelRegistry.method_for(model) in ModelRegistry.NATIVE_CLASSES
        ):
            dataset = inputs[validation]["dataset"]
            media_type = config.get("datasets", {}).get(dataset, {}).get("media_type")
            if media_type == "video":
                fit.setdefault("monitor", "video.auc")
            elif media_type == "image":
                fit.setdefault("monitor", "image.auc")
    evaluation = config.setdefault("evaluation", {})
    if evaluation_inputs is not None:
        if not evaluation_inputs:
            raise ValueError("--test-inputs requires at least one input")
        evaluation["inputs"] = list(evaluation_inputs)
    evaluation.setdefault(
        "inputs", [name for name, item in inputs.items() if item["split"] != "train"]
    )
    names = evaluation["inputs"]
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
        or any(name not in inputs for name in names)
    ):
        raise ValueError("evaluation.inputs must contain unique existing input names")
    if any(inputs[name]["split"] == "train" for name in names):
        raise ValueError("evaluation inputs must use val or test splits")
    threshold = evaluation.setdefault("threshold", 0.0)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
    ):
        raise ValueError("evaluation.threshold must be finite")
    if evaluation.setdefault("video_aggregation", "mean") not in ("mean", "none"):
        raise ValueError("video_aggregation must be mean or none")
    groups = evaluation.setdefault("group_by", ["category"])
    if not isinstance(groups, list) or any(
        not isinstance(g, str) or not g for g in groups
    ):
        raise ValueError("group_by must be a list of metadata field names")
    config.setdefault("seed", 42)
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("device", "auto")
    runtime.setdefault("precision", "fp32")
    runtime.setdefault("log_level", "INFO")
    runtime.setdefault("output_dir", "outputs")
    if runtime["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if "allow_tf32" in runtime and type(runtime["allow_tf32"]) is not bool:
        raise ValueError("runtime.allow_tf32 must be a boolean")
    if runtime["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Invalid log_level")
    resources = ResourceResolver(
        Path(path).resolve().parent, variables=variables, env_file=env_file
    )
    for item in inputs.values():
        # Expand variables when opening a selected input. Anchor plain relative
        # paths here so they resolve against the experiment config.
        if "$" not in item["manifest"]:
            item["manifest"] = str(resources.path(item["manifest"], must_exist=False))
    if require_fit:
        from .runtime import task_factory

        fit = config["fit"]
        registered = {spec.factory for spec in ModelRegistry.SPECS}
        selected_task = (
            task_factory(model)
            if ModelRegistry.resolve_factory(factory) in registered
            else None
        )
        if (
            validate := getattr(selected_task, "validate_fit_config", None)
        ) is not None:
            validate(model, fit, public=public)
        file_fields = getattr(selected_task, "model_file_fields", ())
        if "checkpoint" in fit:
            fit["checkpoint"] = str(resources.path(fit["checkpoint"]))
            fields = getattr(selected_task, "resume_input_fields", ())
            model.update(
                resources.model_settings(
                    {key: model[key] for key in fields if key in model},
                    file_fields=file_fields,
                )
            )
        else:
            config["model"] = resources.model_settings(model, file_fields=file_fields)
        if "ufd" in fit and "checkpoint" not in fit:
            if not isinstance(fit["ufd"], dict):
                raise ValueError("fit.ufd must be a mapping")
            fit["ufd"] = resources.model_settings(fit["ufd"])
    # Outputs may not exist yet; inputs remain the dataset registry's concern.
    runtime["output_dir"] = str(
        resources.path(runtime["output_dir"], must_exist=False, kind="directory")
    )
    return config


def input_settings(config, name):
    return {**config.get("loader", {}), **config["inputs"][name]}
