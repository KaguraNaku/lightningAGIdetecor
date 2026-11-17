"""CPU-only inspection through the same configuration and dataset registry."""
from pathlib import Path
from .config import read_experiment
from .models import ModelRegistry
from .protocol import VideoSamplingPolicy


class ExperimentInspector:
    def inspect(self, config_path, *, datasets=None, env_file=None, model=None):
        from data.config import DataConfig
        from data.manifest import read_manifest, sha256
        config = read_experiment(config_path, env_file=env_file)
        if model is not None:
            config["model"] = ModelRegistry.select(model, config["model"])
        data = DataConfig(datasets, env_file)
        policy = VideoSamplingPolicy(config["video_sampling"]) if "video_sampling" in config else None
        audits = {}
        for name, item in config["inputs"].items():
            value = item["manifest"].replace("${MANIFEST_ROOT}", str(data.manifest_dir()))
            path = Path(data.expand(value)).expanduser()
            header, rows = read_manifest(path)
            if (header["dataset"], header["split"]) != (item["dataset"], item["split"]):
                raise ValueError("Configured input differs from its frozen manifest")
            root = data.dataset(item["dataset"])["root"]
            missing = [str(root / r.path) for r in rows if not (root / r.path).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing input files; first: {missing[0]}")
            audits[name] = {"manifest": str(path), "sha256": sha256(path), "images": len(rows),
                            "sampling": policy.validate(header, rows) if policy else None}
            if (header["split"] == "train" and config["model"]["factory"] == ModelRegistry.get("forada").factory):
                from types import SimpleNamespace
                from detectors.forada.data import ForAdaDataset
                masks = ForAdaDataset(SimpleNamespace(records=rows, header=header, root=root, manifest_path=path),
                                      require_localized=config["model"].get("training_profile") == "author_ffpp_8_32")
                audits[name]["mask_supervision"] = masks.supervision_summary
            if (header["split"] == "train" and config["model"]["factory"] == ModelRegistry.get("prodet").factory):
                from types import SimpleNamespace
                from detectors.prodet.dataset import ProDetDataset
                from detectors.prodet.protocol import neighbor_identity
                neighbors = config["model"].get("training_neighbors")
                if isinstance(neighbors, str):
                    neighbors = neighbor_identity(neighbors)
                source = ProDetDataset(SimpleNamespace(root=root, header=header, records=rows,
                    manifest_path=path), neighbors=neighbors)
                audits[name]["prodet"] = {"data_profile": config["model"]["data_profile"],
                    "quadruples": len(source), "model_views": len(source.records),
                    "neighbors": source.plan.neighbor_identity,
                    "landmarks": len(source.landmark_arrays)}
        return {"status": "validated", "config_sha256": sha256(Path(config_path)),
                "model": ModelRegistry.method_for(config["model"]), "inputs": audits,
                "GPU_forward_executed": False,
                "note": "CPU manifest/identity/resource checks; not a training result or full source rehash"}
