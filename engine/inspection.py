"""CPU-only inspection through the same configuration and dataset registry."""

from pathlib import Path
from types import SimpleNamespace

from .config import read_experiment
from .models import ModelRegistry
from .runtime import task_factory


class ExperimentInspector:
    def inspect(
        self,
        config_path,
        *,
        datasets=None,
        env_file=None,
        model=None,
        batches=0,
        training_input=None,
        validation_input=None,
        evaluation_inputs=None,
        checkpoint=None,
    ):
        if type(batches) is not int or batches < 0:
            raise ValueError("Preflight batches must be a nonnegative integer")
        from data.config import DataConfig
        from data.manifest import read_manifest, sha256

        config = read_experiment(
            config_path,
            env_file=env_file,
            model_name=model,
            training_input=training_input,
            validation_input=validation_input,
            evaluation_inputs=evaluation_inputs,
            checkpoint=checkpoint,
            public=True,
        )
        data = DataConfig.for_experiment(
            config_path, config, registry=datasets, env_file=env_file
        )
        fit = config["fit"]
        selected = dict.fromkeys(
            (fit["input"], fit.get("validation_input"), *config["evaluation"]["inputs"])
        )
        selected.pop(None, None)
        audits, inputs = {}, {}
        for name in selected:
            item = config["inputs"][name]
            path = data.manifest_path(item["manifest"])
            header, rows = read_manifest(path)
            if (header["dataset"], header["split"]) != (item["dataset"], item["split"]):
                raise ValueError("Configured input differs from its manifest")
            root = data.dataset(item["dataset"])["root"]
            inputs[name] = SimpleNamespace(
                root=root, header=header, records=rows, manifest_path=path
            )
            missing = next(
                (row.path for row in rows if not (root / row.path).is_file()), None
            )
            if missing is not None:
                raise FileNotFoundError(f"Missing input file: {root / missing}")
            audits[name] = {
                "manifest": str(path),
                "sha256": sha256(path),
                "images": len(rows),
                "media_type": data.media_type(item["dataset"]),
                "sampling": header["sampling"],
            }
        inspect_training = getattr(
            task_factory(config["model"]), "inspect_training_input", None
        )
        if inspect_training is not None:
            training = fit["input"]
            validation = inputs.get(fit.get("validation_input"))
            audits[training].update(
                inspect_training(config["model"], fit, inputs[training], validation)
            )
        report = {
            "status": "validated",
            "config_sha256": sha256(Path(config_path)),
            "model": ModelRegistry.method_for(config["model"]),
            "inputs": audits,
            "GPU_forward_executed": False,
            "note": "CPU manifest/identity/resource checks; not a training result or full source rehash",
        }
        if batches:
            report["real_batch_checks"] = self.check_batches(
                config, data, selected, batches
            )
        return report

    def check_batches(self, config, data, selected, count):
        """Exercise production Task initialization/data adapters without forward.

        Uses the configured seed, batch size, worker count and collation, in a
        separate CPU process when invoked through the CLI. No optimizer exists.
        """
        import hashlib
        import torch
        from data.loader import create_input_loader
        from .config import input_settings
        from .fit import create_training_task
        from .runtime import (
            Runtime,
            collator_for,
            dataset_adapter_for,
            sampler_factory_for,
        )

        Runtime.create(
            {**config["runtime"], "device": "cpu", "precision": "fp32"}, config["seed"]
        )
        task = create_training_task(config)
        inspect_batch = getattr(task, "inspect_batch", None)
        result = {"model_forward_executed": False, "optimizer_steps": 0, "inputs": {}}
        initialization = getattr(task, "origin", {}).get("pretrained_backbone")
        if initialization is not None:
            result["initialization"] = {
                key: initialization[key]
                for key in ("checkpoint", "checkpoint_sha256")
                if key in initialization
            }
        for name in selected:
            stage = (
                "train"
                if name == config["fit"]["input"]
                else "validation"
                if name == config["fit"].get("validation_input")
                else "evaluation"
            )
            loader = create_input_loader(
                input_settings(config, name),
                config=data,
                collate_fn=collator_for(task, stage),
                dataset_adapter=dataset_adapter_for(task, stage),
                sampler_factory=sampler_factory_for(task, stage),
                seed=config["seed"],
            )
            observed = []
            iterator = iter(loader)
            try:
                for _ in range(min(count, len(loader))):
                    batch = next(iterator)
                    pixels = batch["pixel_values"]
                    if (
                        pixels.dtype != torch.float32
                        or not torch.isfinite(pixels).all()
                    ):
                        raise ValueError(f"Nonfinite/non-FP32 preflight pixels: {name}")
                    row = {
                        "shape": list(pixels.shape),
                        "images": len(pixels),
                        "pixel_sha256": hashlib.sha256(
                            pixels.contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "batch_size": loader.batch_size,
                        "workers": loader.num_workers,
                    }
                    if inspect_batch is not None:
                        row.update(inspect_batch(batch))
                    observed.append(row)
            finally:
                del iterator, loader
            if not observed:
                raise ValueError(f"No real preflight batches available: {name}")
            result["inputs"][name] = observed
        return result
