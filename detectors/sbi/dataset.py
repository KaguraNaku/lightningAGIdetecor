"""Real-only SBI source view with explicit, fingerprinted landmark supervision."""

from dataclasses import replace
import hashlib
import json

import numpy as np

from data.manifest import relative_path, sha256


class SBIRealDataset:
    def __init__(self, dataset):
        self.base = dataset
        self.header = dataset.header
        self.manifest_path = dataset.manifest_path
        self.root = dataset.root
        self.indices, self.records = [], []
        for index, record in enumerate(dataset.records):
            if record.label != 0:
                continue
            descriptor = record.metadata.get("landmarks")
            if descriptor is None:
                if (
                    record.storage != "image"
                    or "frames" not in record.path
                    or not record.path.endswith(".png")
                ):
                    raise ValueError(
                        f"SBI real source {record.id} requires explicit landmarks path/hash/coordinate_size metadata"
                    )
                path = record.path.replace("frames", "landmarks").replace(
                    ".png", ".npy"
                )
                descriptor = {
                    "path": path,
                    "sha256": sha256(self.root / relative_path(path)),
                    "coordinate_size": 256,
                }
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "path",
                "sha256",
                "coordinate_size",
            }:
                raise ValueError(
                    f"SBI landmark metadata for {record.id} requires path/hash/coordinate_size"
                )
            relative_path(descriptor["path"])
            if descriptor["coordinate_size"] != 256:
                raise ValueError(
                    "Configured native SBI landmarks use256-coordinate sidecars; no implicit rescaling"
                )
            if sha256(self.root / descriptor["path"]) != descriptor["sha256"]:
                raise ValueError(f"SBI landmark hash mismatch for {record.id}")
            self.indices.append(index)
            self.records.append(
                replace(
                    record, metadata={**record.metadata, "landmarks": dict(descriptor)}
                )
            )
        if not self.records:
            raise ValueError("SBI training requires at least one real source image")

    def __len__(self):
        return len(self.records)

    def selection_receipt(self):
        sidecars = [
            {"id": record.id, **record.metadata["landmarks"]} for record in self.records
        ]
        encoded = json.dumps(sidecars, sort_keys=True).encode()
        return {
            "policy": "real_label0_before_sampler",
            "parent_manifest": str(self.manifest_path),
            "parent_manifest_sha256": sha256(self.manifest_path),
            "parent_count": len(self.base),
            "selected_count": len(self),
            "excluded_fake_sources": len(self.base) - len(self),
            "sidecars_sha256": hashlib.sha256(encoded).hexdigest(),
            "sidecars": sidecars,
        }

    def __getitem__(self, index):
        sample = self.base[self.indices[index]]
        if sample is None:
            return None
        record = self.records[index]
        descriptor = record.metadata["landmarks"]
        path = self.root / descriptor["path"]
        if sha256(path) != descriptor["sha256"]:
            raise ValueError(f"SBI landmark changed during training: {record.id}")
        values = np.load(path, allow_pickle=False)
        if values.shape not in ((68, 2), (81, 2)) or not np.isfinite(values).all():
            raise ValueError(f"SBI landmark shape/values invalid for {record.id}")
        return {
            **sample,
            "landmarks": values.astype(np.int32),
            "metadata": {**sample["metadata"], "attributes": record.metadata},
        }
