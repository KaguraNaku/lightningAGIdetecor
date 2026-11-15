"""Experiment policy and validation, separate from local dataset root resolution."""

from copy import deepcopy
import math
from pathlib import Path
import re

import yaml

from .resources import ResourceResolver


def read_experiment(path, *, require_fit=True, variables=None, env_file=None):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("Experiment must be a mapping with version: 1")
    config = deepcopy(config)
    for section in ("model", "loader", "fit", "evaluation", "runtime"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    model = config.get("model", {})
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
    evaluation = config.setdefault("evaluation", {})
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
    runtime.setdefault("device", "cpu")
    runtime.setdefault("precision", "fp32")
    runtime.setdefault("log_level", "INFO")
    runtime.setdefault("output_dir", "outputs")
    if runtime["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if "allow_tf32" in runtime and type(runtime["allow_tf32"]) is not bool:
        raise ValueError("runtime.allow_tf32 must be a boolean")
    if runtime["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Invalid log_level")
    resources = ResourceResolver(Path(path).resolve().parent, variables=variables, env_file=env_file)
    for item in inputs.values():
        # Registry variables (including its MANIFEST_ROOT default) are expanded
        # when the selected dataset is opened; simple relative paths can be
        # anchored now without eagerly opening unused datasets.
        if "$" not in item["manifest"]:
            item["manifest"] = str(resources.path(item["manifest"], must_exist=False))
    if require_fit:
        config["model"] = resources.model_settings(config["model"])
        timing = config["fit"].get("gpu_timing")
        if isinstance(timing, dict) and "module" in timing:
            timing["module"] = str(resources.path(timing["module"]))
    # Outputs may not exist yet; inputs remain the dataset registry's concern.
    runtime["output_dir"] = str(resources.path(runtime["output_dir"], must_exist=False, kind="directory"))
    return config


def input_settings(config, name):
    return {**config.get("loader", {}), **config["inputs"][name]}
