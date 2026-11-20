"""CPU-only inspection through the same configuration and dataset registry."""
from pathlib import Path
from .config import read_experiment
from .models import ModelRegistry
from .protocol import VideoSamplingPolicy


class ExperimentInspector:
    def inspect(self, config_path, *, datasets=None, env_file=None, model=None, batches=0):
        if type(batches) is not int or batches < 0:
            raise ValueError("Preflight batches must be a nonnegative integer")
        from data.config import DataConfig
        from data.manifest import read_manifest, sha256
        config = read_experiment(config_path, env_file=env_file)
        if model is not None:
            config["model"] = ModelRegistry.select(model, config["model"])
        data = DataConfig(datasets, env_file)
        policy = VideoSamplingPolicy(config["video_sampling"]) if "video_sampling" in config else None
        ufd = config["fit"].get("ufd") is not None
        audits, source_plans, inputs = {}, {}, {}
        for name, item in config["inputs"].items():
            value = item["manifest"].replace("${MANIFEST_ROOT}", str(data.manifest_dir()))
            path = Path(data.expand(value)).expanduser()
            header, rows = read_manifest(path)
            if (header["dataset"], header["split"]) != (item["dataset"], item["split"]):
                raise ValueError("Configured input differs from its frozen manifest")
            root = data.dataset(item["dataset"])["root"]
            from types import SimpleNamespace
            inputs[name] = SimpleNamespace(root=root, header=header, records=rows, manifest_path=path)
            missing = [str(root / r.path) for r in rows if not (root / r.path).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing input files; first: {missing[0]}")
            audits[name] = {"manifest": str(path), "sha256": sha256(path), "images": len(rows),
                            "sampling": policy.validate(header, rows) if policy else None}
            if (header["split"] == "train" and config["model"]["factory"] == ModelRegistry.get("forada").factory):
                from detectors.forada.data import ForAdaDataset, WholeImageForAdaDataset
                if ufd and name == config["fit"]["input"]:
                    audits[name]["mask_supervision"] = WholeImageForAdaDataset(inputs[name]).selection_receipt()
                else:
                    masks = ForAdaDataset(inputs[name],
                        require_localized=config["model"].get("training_profile") == "author_ffpp_8_32")
                    audits[name]["mask_supervision"] = masks.supervision_summary
            if (header["split"] == "train" and config["model"]["factory"] == ModelRegistry.get("prodet").factory
                    and (not ufd or config["model"].get("data_profile") == "specialized_quadruples")):
                from types import SimpleNamespace
                from detectors.prodet.dataset import ProDetDataset
                from detectors.prodet.protocol import neighbor_identity
                neighbors = config["model"].get("training_neighbors")
                if isinstance(neighbors, str):
                    neighbors = neighbor_identity(neighbors)
                source = ProDetDataset(SimpleNamespace(root=root, header=header, records=rows,
                    manifest_path=path), neighbors=neighbors)
                source_plans[name] = source.plan
                audits[name]["prodet"] = {"data_profile": config["model"]["data_profile"],
                    "quadruples": len(source), "model_views": len(source.records),
                    "neighbors": source.plan.neighbor_identity,
                    "landmarks": len(source.landmark_arrays)}
        if source_plans:
            from detectors.prodet.protocol import validate_source_splits
            validation = config["fit"].get("validation_input")
            if validation not in inputs:
                raise ValueError("ProDet preflight requires the source validation input")
            for name, plan in source_plans.items():
                audits[name]["source_split_validation"] = validate_source_splits(plan, inputs[validation])
        report = {"status": "validated", "config_sha256": sha256(Path(config_path)),
                "model": ModelRegistry.method_for(config["model"]), "inputs": audits,
                "GPU_forward_executed": False,
                "note": "CPU manifest/identity/resource checks; not a training result or full source rehash"}
        if batches:
            report["real_batch_checks"] = self.check_batches(config, data, batches)
        return report

    def check_batches(self, config, data, count):
        """Exercise production Task initialization/data adapters without forward.

        Uses the configured seed, batch size, worker count and collation, in a
        separate CPU process when invoked through the CLI. No optimizer exists.
        """
        import hashlib
        import torch
        from data.loader import create_input_loader
        from .config import input_settings
        from .fit import create_training_task
        from .runtime import Runtime, collator_for, dataset_adapter_for, sampler_factory_for

        Runtime.create({**config["runtime"], "device": "cpu", "precision": "fp32"}, config["seed"])
        task = create_training_task(config)
        result = {"model_forward_executed": False, "optimizer_steps": 0, "inputs": {}}
        initialization = getattr(task, "origin", {}).get("pretrained_backbone")
        if initialization is not None:
            result["initialization"] = {key: initialization[key] for key in ("checkpoint", "checkpoint_sha256") if key in initialization}
        for name, item in config["inputs"].items():
            stage = {"train": "train", "val": "validation", "test": "evaluation"}[item["split"]]
            loader = create_input_loader(input_settings(config, name), config=data,
                collate_fn=collator_for(task, stage), dataset_adapter=dataset_adapter_for(task, stage),
                sampler_factory=sampler_factory_for(task, stage), seed=config["seed"])
            observed = []
            iterator = iter(loader)
            try:
                for _ in range(min(count, len(loader))):
                    batch = next(iterator)
                    pixels = batch["pixel_values"]
                    if pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
                        raise ValueError(f"Nonfinite/non-FP32 preflight pixels: {name}")
                    row = {"shape": list(pixels.shape), "images": len(pixels),
                        "pixel_sha256": hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest(),
                        "batch_size": loader.batch_size, "workers": loader.num_workers}
                    if "prodet_labels" in batch:
                        targets = batch["prodet_labels"]
                        expected = torch.tensor([0., 1., 3., 2.]).repeat_interleave(len(pixels) // 4)
                        if len(pixels) % 4 or targets.dtype != torch.float32 or not torch.equal(targets, expected):
                            raise ValueError("ProDet preflight requires real/SBI/BI/deepfake ordered float domains")
                        row["quadruples"] = len(pixels) // 4
                        row["domain_counts"] = {str(int(v)): int((targets == v).sum()) for v in (0, 1, 3, 2)}
                    observed.append(row)
            finally:
                del iterator, loader
            if not observed:
                raise ValueError(f"No real preflight batches available: {name}")
            result["inputs"][name] = observed
        return result
