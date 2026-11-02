"""Real-only I2G with native sidecar dtypes and bounded first-source replacement."""

from dataclasses import replace
import hashlib
import json
import os

import cv2
import numpy as np
import torch

from data.manifest import relative_path, sha256
from detectors.fwa.generation import rng_state
from .bank import I2GBank
from .generation import I2GGenerator, assemble_pairs


class I2GGenerationError(RuntimeError):
    marker = "I2G_GENERATION_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


def metadata(record):
    return {
        "id": record.id,
        "dataset": record.dataset,
        "split": record.split,
        "video_id": record.video_id,
        "frame_index": record.frame_index,
        "category": record.category,
        "attributes": record.metadata,
    }


class I2GDataset:
    def __init__(self, dataset, bank):
        self.base = dataset
        self.header, self.manifest_path, self.root = (
            dataset.header,
            dataset.manifest_path,
            dataset.root,
        )
        self.bank = I2GBank(bank, dataset)
        self.sidecars = [self._pin_sidecar(r) for r in dataset.records]
        self.indices, self.records, seen = [], [], {}
        self.excluded_fakes, self.excluded_duplicates = 0, 0
        for index, record in enumerate(dataset.records):
            if record.label != 0:
                self.excluded_fakes += 1
                continue
            key = record.storage, record.path, record.row_group, record.row_index
            if key in seen:
                if self.sidecars[index] != self.sidecars[seen[key]]:
                    raise ValueError(
                        "PCL duplicate real frame has conflicting sidecars"
                    )
                self.excluded_duplicates += 1
                continue
            seen[key] = index
            self.indices.append(index)
            self.records.append(
                replace(
                    record,
                    metadata={**record.metadata, "landmarks": self.sidecars[index]},
                )
            )
        if not self.records:
            raise ValueError("PCL needs at least one real training background")
        self._pid, self._generator = None, None

    def _pin_sidecar(self, record):
        descriptor = record.metadata.get("landmarks")
        if descriptor is None:
            if (
                record.storage != "image"
                or "frames" not in record.path
                or not record.path.endswith(".png")
            ):
                raise ValueError(
                    f"PCL {record.id} needs explicit256-coordinate landmarks metadata"
                )
            name = record.path.replace("frames", "landmarks").replace(".png", ".npy")
            descriptor = {"path": name, "coordinate_size": 256}
        if (
            not isinstance(descriptor, dict)
            or set(descriptor)
            not in (
                {"path", "coordinate_size"},
                {"path", "sha256", "coordinate_size"},
            )
            or descriptor["coordinate_size"] != 256
        ):
            raise ValueError(
                "PCL landmarks require relative path and coordinate_size256; no rescaling"
            )
        path = self.root / relative_path(descriptor["path"])
        present = path.exists()
        digest = sha256(path) if present else None
        if "sha256" in descriptor and descriptor["sha256"] != digest:
            raise ValueError(f"PCL sidecar hash/absence mismatch: {record.id}")
        return {
            "path": descriptor["path"],
            "coordinate_size": 256,
            "sha256": digest,
            "bytes": path.stat().st_size if present else None,
        }

    def _landmarks(self, index):
        descriptor = self.sidecars[index]
        path = self.root / descriptor["path"]
        if descriptor["sha256"] is None:
            if path.exists():
                raise ValueError(
                    "PCL previously missing sidecar appeared during training"
                )
            value = np.zeros((81, 2))
        else:
            if (
                path.stat().st_size != descriptor["bytes"]
                or sha256(path) != descriptor["sha256"]
            ):
                raise ValueError("PCL NPY sidecar changed during training")
            value = np.load(path, allow_pickle=False)
        return value, {
            **descriptor,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "native_missing_zero_fill": descriptor["sha256"] is None,
        }

    def _decode(self, index):
        record = self.base.records[index]
        if record.storage == "image":
            path = self.root / record.path
            if not path.exists():
                raise FileNotFoundError(path)
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise ValueError(f"PCL image decode failed: {record.id}")
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        else:
            pixels = np.asarray(self.base._parquet_image(record).convert("RGB"))
        return cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_CUBIC)

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_generator": None}

    def __len__(self):
        return len(self.records)

    def selection_receipt(self):
        encoded = json.dumps(self.sidecars, sort_keys=True).encode()
        return {
            "bank": self.bank.receipt(),
            "parent_count": len(self.base),
            "selected_count": len(self),
            "excluded_fake_sources": self.excluded_fakes,
            "excluded_duplicate_real_frames": self.excluded_duplicates,
            "selected_ids": [r.id for r in self.records],
            "background_policy": "real0 deduplicated by storage identity, stable first manifest occurrence; replaces legacy hash-set order",
            "sidecars": self.sidecars,
            "sidecars_sha256": hashlib.sha256(encoded).hexdigest(),
            "missing_sidecars": sum(s["sha256"] is None for s in self.sidecars),
            "fallback_policy": "native first-source retry once; second decode failure is fatal instead of recursive retries",
            "foreground_inventory": "Complete explicitly bound SAME train manifest; no hidden validation/external neighbor",
        }

    def __getitem__(self, index):
        requested = self.records[index]
        before, attempts, generated = rng_state(), [], None
        try:
            if self._pid != os.getpid():
                self._generator, self._pid = I2GGenerator(), os.getpid()
            for attempt, slot in enumerate((index, 0)):
                actual = self.indices[slot]
                foreground, self_neighbor = self.bank.foreground(actual)
                receipt = {
                    "background": metadata(self.base.records[actual]),
                    "foreground": metadata(self.base.records[foreground]),
                    "rng_after_selection": rng_state(),
                }
                attempts.append(receipt)
                try:
                    background_pixels = self._decode(actual)
                    foreground_pixels = self._decode(foreground)
                except Exception as error:
                    receipt["decode_error"] = f"{type(error).__name__}: {error}"
                    if attempt == 0:
                        continue
                    raise ValueError(
                        "PCL first-source replacement also failed; bounded stop instead of native recursion"
                    ) from error
                break
            points, background_sidecar = self._landmarks(actual)
            _, foreground_sidecar = self._landmarks(
                foreground
            )  # Native unused read still occurs.
            fake, real, mask, generated = self._generator.generate(
                background_pixels, foreground_pixels, points
            )
            if not all(torch.isfinite(v).all() for v in (fake, real, mask)):
                generated["nonfinite_mask_values"] = int((~torch.isfinite(mask)).sum())
                raise ValueError(
                    "PCL nonfinite native generation; raw-mask arithmetic is not repaired"
                )
            after = rng_state()
            return {
                "fake": fake,
                "real": real,
                "mask": mask,
                "label": 0,
                "metadata": metadata(requested),
                "rng_before": before,
                "rng_after": after,
                "generation": {
                    **generated,
                    "source_label": 0,
                    "generated_label": 1,
                    "fallback": False,
                    "source_replacement": len(attempts) > 1,
                    "requested_background_id": requested.id,
                    "actual_background": metadata(self.base.records[actual]),
                    "foreground": metadata(self.base.records[foreground]),
                    "neighbor_fallback_to_self": self_neighbor,
                    "background_sidecar": background_sidecar,
                    "foreground_sidecar": foreground_sidecar,
                    "decode_attempts": attempts,
                },
            }
        except Exception as error:
            raise I2GGenerationError(
                {
                    "requested_background": metadata(requested),
                    "bank": self.bank.identity,
                    "attempts": attempts,
                    "generation": generated,
                    "error": f"{type(error).__name__}: {error}",
                    "rng_before": before,
                    "rng_after": rng_state(),
                }
            ) from error


class I2GPairCollator:
    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError("PCL cannot silently skip a requested source")
        batch = assemble_pairs([(s["fake"], s["real"], s["mask"]) for s in samples])
        batch["metadata"] = [
            {
                **s["metadata"],
                "view": kind,
                "actual_background_id": s["generation"]["actual_background"]["id"],
            }
            for kind in ("original", "generated")
            for s in samples
        ]
        batch["sources"] = [
            {"metadata": s["metadata"], "label": 0, "generation": s["generation"]}
            for s in samples
        ]
        batch["generation_rng"] = {
            "before": samples[0]["rng_before"],
            "after": samples[-1]["rng_after"],
            "collation_consumes_rng": False,
        }
        return batch
