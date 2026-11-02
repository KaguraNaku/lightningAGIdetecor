"""Explicit repair of broken native tuple/rank/dtype transport, with pinned masks."""

from dataclasses import replace
import json
import os

import cv2
import numpy as np
import torch

from data.manifest import relative_path, sha256
from detectors._shared.transforms import build_augmentation
from detectors.fwa.generation import rng_state
from .preprocessing import frequency, normalized_rgb


class LRLDataError(RuntimeError):
    marker = "LRL_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


class LRLDataset:
    def __init__(self, dataset, preprocessing, *, training):
        self.base, self.preprocessing, self.training = (
            dataset,
            dict(preprocessing),
            training,
        )
        self.header, self.manifest_path, self.root = (
            dataset.header,
            dataset.manifest_path,
            dataset.root,
        )
        self.records = [
            replace(r, metadata={**r.metadata, "mask": self._pin_mask(r)})
            for r in dataset.records
        ]
        self._pid, self._augmentation = None, None

    def _pin_mask(self, record):
        descriptor = record.metadata.get("mask")
        if descriptor is None:
            path = (
                record.path.replace("frames", "masks")
                if record.storage == "image" and "frames" in record.path
                else None
            )
            descriptor = {"path": path}
        if not isinstance(descriptor, dict) or set(descriptor) not in (
            {"path"},
            {"path", "sha256"},
        ):
            raise ValueError(
                "LRL mask descriptor requires relative path and optional sha256"
            )
        path = (
            self.root / relative_path(descriptor["path"])
            if descriptor["path"] is not None
            else None
        )
        present = path is not None and path.exists()
        digest = sha256(path) if present else None
        if "sha256" in descriptor and descriptor["sha256"] != digest:
            raise ValueError("LRL mask hash/absence mismatch")
        return {
            "path": descriptor["path"],
            "sha256": digest,
            "bytes": path.stat().st_size if present else None,
        }

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_augmentation": None}

    def __len__(self):
        return len(self.records)

    def _decode(self, index):
        record = self.records[index]
        if record.storage == "image":
            path = self.root / record.path
            if not path.exists():
                raise FileNotFoundError(path)
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise ValueError(f"LRL could not decode {record.id}")
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        else:
            pixels = np.asarray(self.base._parquet_image(record).convert("RGB"))
        return cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_CUBIC)

    def _mask(self, index):
        descriptor = self.records[index].metadata["mask"]
        path = (
            self.root / descriptor["path"] if descriptor["path"] is not None else None
        )
        decoded_zero = False
        if descriptor["sha256"] is None:
            if path is not None and path.exists():
                raise ValueError("LRL previously missing mask appeared during run")
            value = np.zeros((256, 256, 1))  # Native missing boundary is float64.
        else:
            if (
                path.stat().st_size != descriptor["bytes"]
                or sha256(path) != descriptor["sha256"]
            ):
                raise ValueError("LRL pinned mask changed during run")
            value = cv2.imread(str(path), 0)
            if value is None:
                value, decoded_zero = np.zeros((256, 256)), True
            value = np.float32(
                np.expand_dims(cv2.resize(value, (256, 256)) / 255, axis=2)
            )
        return value, {
            **descriptor,
            "native_dtype": str(value.dtype),
            "native_missing_zeros": descriptor["sha256"] is None,
            "native_decode_zero_fallback": decoded_zero,
        }

    def selection_receipt(self):
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256(self.manifest_path),
            "samples": len(self),
            "policy": "All explicit manifest rows; no replacement, hidden subset or label change",
            "masks": [{"id": r.id, **r.metadata["mask"]} for r in self.records],
            "transport_repairs": [
                "Named fields replace broken4-from5 tuple unpack",
                "HW/HWC1 grayscale maps become1HW directly",
                "Native float64 missing zeros cast to float32 for BCE",
            ],
            "decode_failure_policy": "Diagnostic fatal error; replaces broken legacy recursive data path rather than silently skipping/replacing sources",
            "original_full_data_equivalence_claim": False,
        }

    def __getitem__(self, index):
        record, before = self.records[index], rng_state()
        try:
            if self._pid != os.getpid():
                self._augmentation = (
                    build_augmentation(self.preprocessing["data_aug"])
                    if self.training
                    else None
                )
                self._pid = os.getpid()
            pixels = self._decode(index)
            mask, mask_receipt = self._mask(index)
            mask = mask.squeeze(2)
            if self._augmentation is not None:
                kwargs = {"image": pixels}
                if mask.max() > 0:
                    kwargs["mask"] = mask
                transformed = self._augmentation(**kwargs)
                pixels, mask = transformed["image"], transformed.get("mask", mask)
            return {
                "pixel_values": normalized_rgb(pixels),
                "idct": frequency(pixels),
                "mask": torch.from_numpy(
                    np.ascontiguousarray(mask.astype(np.float32)[None])
                ),
                "label": record.label,
                "metadata": {
                    "id": record.id,
                    "dataset": record.dataset,
                    "split": record.split,
                    "video_id": record.video_id,
                    "frame_index": record.frame_index,
                    "category": record.category,
                    "attributes": record.metadata,
                },
                "transport": {
                    "mask": mask_receipt,
                    "training_augmentation": self.training,
                    "mask_transport_dtype": "float32",
                    "mask_transport_shape": [1, 256, 256],
                    "rng_before": before,
                    "rng_after": rng_state(),
                },
            }
        except Exception as error:
            raise LRLDataError(
                {
                    "source_id": record.id,
                    "label": record.label,
                    "mask": record.metadata["mask"],
                    "error": f"{type(error).__name__}: {error}",
                    "rng_before": before,
                    "rng_after": rng_state(),
                }
            ) from error


class LRLCollator:
    def __call__(self, samples):
        if not samples or any(s is None for s in samples):
            raise ValueError("LRL cannot silently drop a required RGB/IDCT/mask source")
        return {
            "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
            "idct": torch.stack([s["idct"] for s in samples]),
            "masks": torch.stack([s["mask"] for s in samples]),
            "labels": torch.tensor([s["label"] for s in samples]),
            "metadata": [s["metadata"] for s in samples],
            "transport": [s["transport"] for s in samples],
        }
