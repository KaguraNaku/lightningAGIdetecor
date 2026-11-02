"""Thin CLI assembly for fit, evaluate and complete experiment runs."""

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import platform

import torch
import yaml

from data.config import DataConfig
from data.loader import create_input_loader
from data.manifest import sha256

from .config import input_settings, read_experiment
from .evaluate import evaluate
from .fit import fit, restore
from .runtime import (
    Runtime,
    release_cpu_memory,
    atomic_save,
    configure_logging,
    collator_for,
    dataset_adapter_for,
    sampler_factory_for,
    create_run_dir,
    create_task,
)

logger = logging.getLogger(__name__)


def execute(
    command,
    config_path,
    *,
    checkpoint=None,
    datasets=None,
    env_file=None,
    overrides=None,
):
    config = read_experiment(config_path, require_fit=command != "evaluate")
    for key, value in (overrides or {}).items():
        if value is not None:
            config["runtime"][key] = str(value)
    if command not in ("fit", "evaluate", "run"):
        raise ValueError(f"Unknown command: {command}")
    if command == "evaluate" and checkpoint is None:
        raise ValueError("evaluate requires --checkpoint")
    directory = create_run_dir(config["runtime"]["output_dir"], command)
    root_logger, handlers, previous_level = configure_logging(
        directory, config["runtime"]["log_level"]
    )
    status = {
        "status": "running",
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "seed": config["seed"],
        "inputs": {},
    }
    atomic_save(directory / "run.json", status)
    try:
        (directory / "resolved-config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        runtime = Runtime.create(config["runtime"], config["seed"])
        data = DataConfig(datasets, env_file)
        status["dataset_registry_sha256"] = sha256(data.path)
        logger.info("Run directory: %s", directory)
        names = config["evaluation"]["inputs"] if command != "fit" else []
        if command != "fit" and not names:
            raise ValueError("No evaluation inputs configured")
        if command == "evaluate":
            task, saved = restore(checkpoint)
            # Model construction and score policy belong to the saved model, not test data.
            config["model"] = saved["model"]
            config["evaluation"]["threshold"] = saved["score"]["threshold"]
            status["checkpoint"] = {
                "path": str(Path(checkpoint).resolve()),
                "sha256": sha256(Path(checkpoint)),
            }
        else:
            task = create_task(config["model"])
        (directory / "resolved-config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )

        def make_loader(name, stage="evaluation"):
            loader = create_input_loader(
                input_settings(config, name),
                config=data,
                collate_fn=collator_for(task, stage),
                dataset_adapter=dataset_adapter_for(task, stage),
                sampler_factory=sampler_factory_for(task, stage),
                seed=config["seed"],
            )
            status["inputs"][name] = {
                "manifest_sha256": sha256(loader.dataset.manifest_path),
                "dataset": loader.dataset.header["dataset"],
                "split": loader.dataset.header["split"],
                "expected": len(loader.dataset),
            }
            atomic_save(directory / "run.json", status)
            return loader

        if command != "evaluate":
            name = config["fit"]["input"]
            loader = make_loader(name, "train")
            validation_name = config["fit"].get("validation_input")
            validation_loader = (
                make_loader(validation_name, "validation") if validation_name else None
            )
            fit_dir = directory / "fit"
            fit_dir.mkdir()
            path, status["fit"] = fit(
                task,
                loader,
                validation_loader=validation_loader,
                runtime=runtime,
                config=config,
                output_dir=fit_dir,
                provenance={"seed": config["seed"], "inputs": dict(status["inputs"])},
            )
            del loader, validation_loader
            status["checkpoint"] = {"path": str(path), "sha256": sha256(path)}
            if command == "run":
                del task
                release_cpu_memory()
                task, _ = restore(path)
        reports = {}
        for name in names:
            loader = make_loader(name)
            reports[name] = evaluate(
                task,
                loader,
                runtime=runtime,
                settings=config["evaluation"],
                output_dir=directory / name,
            )
            del loader
        if reports:
            temporary = directory / "summary.partial.csv"
            with temporary.open("w", newline="") as stream:
                rows = []
                for name, report in reports.items():
                    row = {"input": name, **report["coverage"]}
                    metric_names = dict.fromkeys(
                        (*report["image"], *(report["video"] or {}))
                    )
                    for unit in ("image", "video"):
                        metrics = report[unit] or {}
                        row.update(
                            {f"{unit}_{key}": metrics.get(key) for key in metric_names}
                        )
                    rows.append(row)
                fields = list(dict.fromkeys(k for row in rows for k in row))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(directory / "summary.csv")
        status.update(
            status="complete",
            evaluation={name: report["coverage"] for name, report in reports.items()},
        )
        return directory
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        logger.exception("Experiment failed")
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_save(directory / "run.json", status)
        for handler in handlers:
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(previous_level)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run model-independent image detection experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "evaluate", "run"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config", type=Path, required=True, help="Experiment YAML"
        )
        command.add_argument("--datasets", type=Path, help="Dataset registry YAML")
        command.add_argument("--env-file", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--device")
        command.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
        command.add_argument(
            "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        directory = execute(
            args.command,
            args.config,
            checkpoint=getattr(args, "checkpoint", None),
            datasets=args.datasets,
            env_file=args.env_file,
            overrides={
                key: getattr(args, key)
                for key in ("output_dir", "device", "precision", "log_level")
            },
        )
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(directory)
