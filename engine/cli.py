"""Thin CLI assembly for fit, evaluate and complete experiment runs."""

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import platform

from .models import ModelRegistry

logger = logging.getLogger(__name__)


def execute(
    command,
    config_path,
    *,
    checkpoint=None,
    datasets=None,
    env_file=None,
    overrides=None,
    model_name=None,
    training_input=None,
    validation_input=None,
    evaluation_inputs=None,
    timing=False,
    public=False,
):
    import torch
    import yaml

    from data.config import DataConfig
    from data.loader import create_input_loader
    from data.manifest import sha256

    from .accelerator import set_tf32, tf32_flags
    from .config import input_settings, read_experiment
    from .evaluate import evaluate
    from .fit import CheckpointReference, create_training_task, fit, restore
    from .summary import summary_rows
    from .runtime import (
        Runtime,
        release_cpu_memory,
        atomic_save,
        configure_logging,
        collator_for,
        dataset_adapter_for,
        sampler_factory_for,
        create_run_dir,
    )
    config = read_experiment(
        config_path, require_fit=command != "evaluate", env_file=env_file, model_name=model_name,
        training_input=training_input, validation_input=validation_input,
        evaluation_inputs=evaluation_inputs, checkpoint=checkpoint, public=public,
    )
    for key, value in (overrides or {}).items():
        if value is not None:
            config["runtime"][key] = value if key == "allow_tf32" else str(value)
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
    previous_tf32 = None
    runtime = None
    try:
        (directory / "resolved-config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        runtime = Runtime.create(config["runtime"], config["seed"], timing=timing)
        data = DataConfig.for_experiment(config_path, config, registry=datasets, env_file=env_file)
        status["dataset_registry_sha256"] = sha256(data.path)
        logger.info("Run directory: %s", directory)
        names = config["evaluation"]["inputs"] if command != "fit" else []
        if command != "fit" and not names:
            raise ValueError("No evaluation inputs configured")
        if command == "evaluate":
            task, saved = restore(checkpoint)
            # Model construction and score policy belong to the saved model, not test data.
            config["model"] = saved["model"]
            if model_name is not None:
                ModelRegistry.select(model_name, config["model"])
            config["evaluation"]["threshold"] = saved["score"]["threshold"]
            status["checkpoint"] = CheckpointReference(checkpoint).state_dict()
        else:
            task = create_training_task(config)
        default_tf32 = getattr(task, "default_allow_tf32", None)
        if runtime.device.type == "cuda" and "allow_tf32" not in config["runtime"] and default_tf32 is not None:
            previous_tf32 = tf32_flags()
            set_tf32(default_tf32, default_tf32)
            config["runtime"]["allow_tf32"] = default_tf32
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
                "media_type": loader.media_type,
                "sampling": loader.dataset.header["sampling"],
                "expected": len(loader.dataset),
            }
            atomic_save(directory / "run.json", status)
            return loader

        if command != "evaluate":
            with runtime.time_region("fit"):
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
            phase = f"evaluate/{name}"
            with runtime.time_region(phase):
                loader = make_loader(name)
                reports[name] = evaluate(
                    task,
                    loader,
                    runtime=runtime,
                    settings=config["evaluation"],
                    output_dir=directory / name,
                    phase=phase,
                )
            del loader
        if reports:
            temporary = directory / "summary.partial.csv"
            with temporary.open("w", newline="") as stream:
                rows = summary_rows(reports)
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
        if previous_tf32 is not None:
            set_tf32(*previous_tf32)
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_save(directory / "run.json", status)
        if runtime is not None and runtime.timer is not None:
            runtime.timer.metadata.update(command=command, status=status["status"], precision=runtime.precision)
            runtime.timer.write(directory / "timing.json")
        for handler in handlers:
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(previous_level)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("prepare-data", "check-data"):
        from data.cli import prepare_main, check_main
        return {"prepare-data": prepare_main, "check-data": check_main}[argv[0]](argv[1:])
    parser = argparse.ArgumentParser(
        description="Run model-independent image detection experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List the nine models without loading any backend")
    commands.add_parser("prepare-data", help="Prepare frozen manifests; forwards to the shared data module", add_help=False)
    commands.add_parser("check-data", help="Decode manifest inputs using the shared data module", add_help=False)
    commands.add_parser("environment", help="Report installed dependency versions without model execution")
    importer = commands.add_parser("import-checkpoint", help="Wrap complete native weights for ordinary evaluate; no training")
    importer.add_argument("--model", choices=tuple(ModelRegistry.NATIVE_CLASSES))
    importer.add_argument("--config", type=Path, help="Defaults to configs/config.yaml")
    importer.add_argument("--env-file", type=Path)
    importer.add_argument("--input", required=True, type=Path, help="Original complete detector checkpoint")
    importer.add_argument("--output", required=True, type=Path, help="New engine checkpoint; existing files are preserved")
    for name in ("fit", "evaluate", "run", "check"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config", type=Path, help="Defaults to configs/config.yaml"
        )
        command.add_argument("--model", choices=tuple(spec.name for spec in ModelRegistry.SPECS))
        if name != "evaluate":
            command.add_argument("--train-input", help="Configured source train input")
            command.add_argument("--validation-input", help="Configured source validation input")
        command.add_argument("--test-inputs", nargs="+", help="Configured val/test input names")
        command.add_argument("--datasets", type=Path, help="Dataset registry YAML")
        command.add_argument("--env-file", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--device", help="cpu, auto, or an available PyTorch accelerator such as cuda:0")
        command.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
        if name != "check":
            command.add_argument(
                "--tf32", dest="allow_tf32", action=argparse.BooleanOptionalAction,
                help="Allow TF32 on CUDA",
            )
            command.add_argument(
                "--timing", action=argparse.BooleanOptionalAction, default=False,
                help="Write a timing report for the command (default: off)",
            )
        command.add_argument(
            "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        command.add_argument("--checkpoint", type=Path, required=name == "evaluate",
            help="Fitted model to evaluate" if name == "evaluate" else "Continue from a fitted model.pth")
        if name == "check":
            command.add_argument("--batches", type=int, default=0,
                help="Also initialize the selected Task and decode this many real batches per input on CPU; no model forward or training")
    args = parser.parse_args(argv)
    if args.command == "environment":
        import importlib.metadata
        import json
        import platform
        versions = {}
        for package in ("torch", "torchvision", "transformers", "Pillow", "numpy", "scipy", "albumentations"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        print(json.dumps({"python": platform.python_version(), "packages": versions,
                          "GPU_forward_executed": False}, indent=2))
        return
    if args.command == "models":
        import json
        print(json.dumps({"models":ModelRegistry.describe()},indent=2))
        return
    if args.config is None:
        if args.model is None:
            parser.error('Specify --model or --config')
        args.config = ModelRegistry.default_config(args.model)
    try:
        if args.command == "import-checkpoint":
            from .fit import CheckpointImporter
            converted = CheckpointImporter(args.config, model_name=args.model, env_file=args.env_file).convert(
                args.input, args.output
            )
            print(converted)
            return
        if args.command == "check":
            import json
            from .inspection import ExperimentInspector
            print(json.dumps(ExperimentInspector().inspect(args.config, datasets=args.datasets,
                env_file=args.env_file, model=args.model, batches=args.batches,
                training_input=args.train_input, validation_input=args.validation_input,
                evaluation_inputs=args.test_inputs, checkpoint=args.checkpoint), indent=2))
            return
        directory = execute(
            args.command,
            args.config,
            checkpoint=getattr(args, "checkpoint", None),
            datasets=args.datasets,
            env_file=args.env_file,
            model_name=args.model,
            training_input=getattr(args, "train_input", None),
            validation_input=getattr(args, "validation_input", None),
            evaluation_inputs=args.test_inputs,
            timing=args.timing,
            public=True,
            overrides={
                key: getattr(args, key)
                for key in ("output_dir", "device", "precision", "allow_tf32", "log_level")
            },
        )
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(directory)


if __name__ == "__main__":
    main()
