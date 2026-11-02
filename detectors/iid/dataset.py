"""Native identity labels/augmentation with explicit requested/actual source IDs."""

from dataclasses import replace
import json
import os
from pathlib import PurePosixPath

import cv2
import numpy as np
from PIL import Image
import torch

from data.manifest import sha256
from detectors._shared.transforms import ImageTransform
from detectors.fwa.generation import rng_state


class IIDSampleError(RuntimeError):
    marker = "IID_SAMPLE_FAILURE:"

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
        "identity_index": record.metadata["identity_index"],
    }


class IIDDataset:
    def __init__(self, dataset, preprocessing):
        if dataset.header["split"] != "train":
            raise ValueError("IID identity dataset is a train-only adapter")
        self.base, self.preprocessing = dataset, dict(preprocessing)
        self.header, self.manifest_path, self.root = (
            dataset.header,
            dataset.manifest_path,
            dataset.root,
        )
        self.records, self.identity_sources = [], []
        for record in dataset.records:
            value = record.metadata.get("identity_index")
            source = "explicit_manifest_identity_index"
            if value is None:
                if record.storage != "image":
                    raise ValueError(
                        "IID Parquet rows require explicit identity_index metadata"
                    )
                try:
                    value = int(PurePosixPath(record.path).parent.name.split("_")[-1])
                except ValueError as error:
                    raise ValueError(
                        f"IID {record.id} requires native numeric parent-folder identity or explicit identity_index"
                    ) from error
                source = "native_parent_folder_last_underscore_segment"
            if type(value) is not int or not 0 <= value < 1000:
                raise ValueError("IID identity_index must be an integer in[0,1000)")
            self.records.append(
                replace(record, metadata={**record.metadata, "identity_index": value})
            )
            self.identity_sources.append(source)
        self._pid, self._transform = None, None

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_transform": None}

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
                raise ValueError(f"IID image decode failed: {record.id}")
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        else:
            pixels = np.asarray(self.base._parquet_image(record).convert("RGB"))
        return Image.fromarray(
            cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_CUBIC)
        )

    def selection_receipt(self):
        return {
            "policy": "All explicit train rows in manifest order; no class-balancing/resampling or identity remapping",
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256(self.manifest_path),
            "identities": [
                {
                    "id": r.id,
                    "label": r.label,
                    "identity_index": r.metadata["identity_index"],
                    "source": s,
                }
                for r, s in zip(self.records, self.identity_sources, strict=True)
            ],
            "fallback": "Native index0 replacement with its actual identity AND label, once; second decode failure fatal instead of recursion",
            "one_class_batch": "Native EIC is undefined; preserve objective and fail explicitly, no balance/mask repair",
            "Parquet_transport": "Explicit identity metadata replaces unsupported legacy tuple-path parsing",
        }

    def __getitem__(self, index):
        requested, attempts, before = self.records[index], [], rng_state()
        try:
            if self._pid != os.getpid():
                self._transform, self._pid = (
                    ImageTransform(self.preprocessing, training=True),
                    os.getpid(),
                )
            for attempt, actual in enumerate((index, 0)):
                record = self.records[actual]
                receipt = {"source": metadata(record), "label": record.label}
                attempts.append(receipt)
                try:
                    image = self._decode(actual)
                except Exception as error:
                    receipt["decode_error"] = f"{type(error).__name__}: {error}"
                    if attempt == 0:
                        continue
                    raise ValueError(
                        "IID first-source replacement also failed; bounded stop instead of recursive retries"
                    ) from error
                break
            pixels = self._transform(image)
            return {
                "pixel_values": pixels,
                "label": record.label,
                "id_index": record.metadata["identity_index"],
                "metadata": metadata(record),
                "requested": metadata(requested),
                "requested_label": requested.label,
                "replacement": len(attempts) > 1,
                "attempts": attempts,
                "rng_before": before,
                "rng_after": rng_state(),
            }
        except Exception as error:
            raise IIDSampleError(
                {
                    "requested": metadata(requested),
                    "attempts": attempts,
                    "error": f"{type(error).__name__}: {error}",
                    "rng_before": before,
                    "rng_after": rng_state(),
                }
            ) from error


class IIDCollator:
    def __call__(self, samples):
        if not samples or any(s is None for s in samples):
            raise ValueError("IID cannot silently skip a requested identity source")
        return {
            "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
            "labels": torch.tensor([s["label"] for s in samples]),
            "id_indices": torch.tensor([s["id_index"] for s in samples]),
            "metadata": [s["metadata"] for s in samples],
            "requests": [
                {
                    "metadata": s["requested"],
                    "label": s["requested_label"],
                    "replacement": s["replacement"],
                    "attempts": s["attempts"],
                    "rng_before": s["rng_before"],
                    "rng_after": s["rng_after"],
                }
                for s in samples
            ],
        }
