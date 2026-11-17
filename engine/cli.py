"""Thin CLI assembly for fit, evaluate and complete experiment runs."""

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
import time

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
    evaluation_inputs=None,
):
    import torch
    import yaml

    from data.config import DataConfig
    from data.loader import create_input_loader
    from data.manifest import sha256

    from .config import input_settings, read_experiment
    from .evaluate import evaluate
    from .fit import CheckpointReference, fit, restore
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
        create_task,
    )
    started_clock = time.perf_counter()
    config = read_experiment(config_path, require_fit=command != "evaluate", env_file=env_file)
    if model_name is not None:
        config["model"] = ModelRegistry.select(model_name, config.get("model", {}))
    if training_input is not None:
        if training_input not in config["inputs"] or config["inputs"][training_input]["split"] != "train":
            raise ValueError("--train-input must name a configured source train input")
        config["fit"]["input"] = training_input
    if evaluation_inputs is not None:
        if (not evaluation_inputs or len(set(evaluation_inputs)) != len(evaluation_inputs)
                or any(name not in config["inputs"] or config["inputs"][name]["split"] == "train"
                       for name in evaluation_inputs)):
            raise ValueError("--test-inputs must name unique configured validation/test inputs")
        config["evaluation"]["inputs"] = list(evaluation_inputs)
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
            if model_name is not None:
                ModelRegistry.select(model_name, config["model"])
            config["evaluation"]["threshold"] = saved["score"]["threshold"]
            status["checkpoint"] = CheckpointReference(checkpoint).state_dict()
        else:
            task = create_task(config["model"])
        (directory / "resolved-config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )

        def make_loader(name, stage="evaluation"):
            from .protocol import VideoSamplingPolicy
            policy = config.get("video_sampling")
            def adapt(dataset):
                if policy is not None:
                    audit = VideoSamplingPolicy(policy).validate(dataset.header, dataset.records)
                    status.setdefault("sampling", {})[name] = audit
                adapter = dataset_adapter_for(task, stage)
                return adapter(dataset) if adapter is not None else dataset
            loader = create_input_loader(
                input_settings(config, name),
                config=data,
                collate_fn=collator_for(task, stage),
                dataset_adapter=adapt,
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
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_save(directory / "run.json", status)
        timing_path = directory / 'fit/timing.json'
        if timing_path.exists():
            import json
            timing = json.loads(timing_path.read_text())
            timing['metadata'].update(end_to_end_wall_seconds=time.perf_counter() - started_clock,
                end_to_end_scope='engine.cli.execute entry to finally;includes model setup,data,fit,source validation,checkpoints;excludes Python process imports',
                run_id=directory.name)
            atomic_save(timing_path, timing)
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
    importer.add_argument("--config", type=Path, help="Defaults to the selected model's maintained preset")
    importer.add_argument("--env-file", type=Path)
    importer.add_argument("--input", required=True, type=Path, help="Original complete detector checkpoint")
    importer.add_argument("--output", required=True, type=Path, help="New engine checkpoint; existing files are preserved")
    importer.add_argument("--legacy-weight-mode", choices=["merged", "unmerged"])
    for name in ("fit", "evaluate", "run", "check"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config", type=Path, help="Experiment YAML; defaults to the selected model's maintained preset"
        )
        command.add_argument("--model", choices=tuple(spec.name for spec in ModelRegistry.SPECS))
        command.add_argument("--train-input", help="Configured source train input")
        command.add_argument("--test-inputs", nargs="+", help="Configured val/test input names")
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
        # The public CLI exposes only the nine supported factories. The low-level
        # execute() API retains injectable tasks for isolated engine tests.
        import yaml
        declaration = yaml.safe_load(args.config.read_text())
        if not isinstance(declaration, dict) or not isinstance(declaration.get('model'), dict):
            raise ValueError('A supported model declaration is required')
        ModelRegistry.validate_factory(declaration['model'])
        if args.command == "import-checkpoint":
            from .fit import CheckpointImporter
            converted = CheckpointImporter(args.config, model_name=args.model, env_file=args.env_file,
                legacy_weight_mode=args.legacy_weight_mode).convert(args.input, args.output)
            print(converted)
            return
        if args.command != "evaluate" and ModelRegistry.method_for(declaration["model"]) in ("gend", "forada", "prodet"):
            from .protocol import VideoSamplingPolicy
            VideoSamplingPolicy(declaration.get("video_sampling"))
            if (ModelRegistry.resolve_factory(declaration["model"].get("factory")) == ModelRegistry.get("prodet").factory
                    and declaration["model"].get("data_profile") != "specialized_quadruples"):
                raise ValueError("New ProDet training requires author specialized_quadruples; legacy checkpoints remain evaluable")
        if (args.command != 'evaluate'
                and ModelRegistry.resolve_factory(declaration['model'].get('factory')) == ModelRegistry.LEGACY_LIGHTNING):
            raise ValueError('New Lightning training requires the formal fixed-eight Task')
        if args.command == "check":
            import json
            from .inspection import ExperimentInspector
            print(json.dumps(ExperimentInspector().inspect(args.config, datasets=args.datasets,
                env_file=args.env_file, model=args.model), indent=2))
            return
        directory = execute(
            args.command,
            args.config,
            checkpoint=getattr(args, "checkpoint", None),
            datasets=args.datasets,
            env_file=args.env_file,
            model_name=args.model,
            training_input=args.train_input,
            evaluation_inputs=args.test_inputs,
            overrides={
                key: getattr(args, key)
                for key in ("output_dir", "device", "precision", "log_level")
            },
        )
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(directory)


if __name__ == "__main__":
    main()
