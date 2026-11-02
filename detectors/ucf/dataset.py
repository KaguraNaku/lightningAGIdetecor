"""Fake-indexed native pairing, sampled real provenance and specific labels."""

from dataclasses import replace
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
import torch

from data.manifest import sha256
from detectors._shared.transforms import ImageTransform
from detectors.fwa.generation import rng_state


DEFAULT_LABEL_MAP = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 2, "FF-FS": 3, "FF-NT": 4}


class UCFDataError(RuntimeError):
    marker = "UCF_DATA_FAILURE:"

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
        "specific_label": record.metadata["specific_label"],
    }


class UCFDataset:
    def __init__(self, dataset, preprocessing, *, specific_classes=5, label_map=None):
        if dataset.header["split"] != "train":
            raise ValueError("UCF pairing is a train-only adapter")
        self.base, self.preprocessing = dataset, dict(preprocessing)
        self.header, self.manifest_path, self.root = (
            dataset.header,
            dataset.manifest_path,
            dataset.root,
        )
        self.specific_classes = specific_classes
        self.label_map = dict(DEFAULT_LABEL_MAP if label_map is None else label_map)
        if (
            type(specific_classes) is not int
            or specific_classes < 2
            or any(
                not isinstance(key, str)
                or type(value) is not int
                or not 0 <= value < specific_classes
                for key, value in self.label_map.items()
            )
        ):
            raise ValueError(
                "UCF needs a valid specific class count and explicit label map"
            )
        self.all_records = []
        for row in dataset.records:
            if "specific_label" in row.metadata:
                value, source = row.metadata["specific_label"], "explicit_metadata"
            elif row.label == 0:
                value, source = 0, "native_real_zero"
            else:
                value, source = (
                    self.label_map.get(row.category),
                    "declared_category_map",
                )
            if (
                type(value) is not int
                or not 0 <= value < specific_classes
                or (value == 0) != (row.label == 0)
            ):
                raise ValueError(
                    f"UCF {row.id} requires specific_label0 for real or1..{specific_classes - 1} for fake; supply metadata or an explicit category map"
                )
            self.all_records.append(
                replace(
                    row,
                    metadata={
                        **row.metadata,
                        "specific_label": value,
                        "specific_label_source": source,
                    },
                )
            )
        self.records = [row for row in self.all_records if row.label == 1]
        self.real_records = [row for row in self.all_records if row.label == 0]
        if not self.records or not self.real_records:
            raise ValueError("UCF requires both indexed fake and sampled real sources")
        self._pid, self._transform = None, None

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_transform": None}

    def __len__(self):
        return len(self.records)

    def _decode(self, row):
        if row.storage == "image":
            path = self.root / row.path
            if not path.exists():
                raise FileNotFoundError(path)
            value = cv2.imread(str(path))
            if value is None:
                raise ValueError(f"UCF failed to decode {row.id}")
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
        else:
            value = np.asarray(self.base._parquet_image(row).convert("RGB"))
        return Image.fromarray(
            cv2.resize(value, (256, 256), interpolation=cv2.INTER_CUBIC)
        )

    def selection_receipt(self):
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256(self.manifest_path),
            "policy": "All fake rows indexed in stable manifest order; real pool sampled with replacement before IO. No silent class remapping, balancing or decode replacement.",
            "label_map": self.label_map,
            "specific_classes": self.specific_classes,
            "sources": [
                {
                    "id": row.id,
                    "label": row.label,
                    "specific_label": row.metadata["specific_label"],
                    "binding": row.metadata["specific_label_source"],
                }
                for row in self.all_records
            ],
            "indexed_fake_count": len(self.records),
            "sampled_real_pool_count": len(self.real_records),
            "order": "sample real, decode fake then real, augment fake then real; collate all real then all fake",
            "Parquet_transport": "Explicit manifest records replace unsupported native tuple-path .replace; RGB decoding/augmentation remains equivalent after decoded uint8 pixels",
            "evaluation": "Ordinary unpaired native RGB256 preprocessing; no specific labels required",
        }

    def __getitem__(self, index):
        fake, before, real = self.records[index], rng_state(), None
        try:
            # This draw precedes decoder IO and augmentation in the native body.
            real = self.real_records[random.randint(0, len(self.real_records) - 1)]
            fake_image, real_image = self._decode(fake), self._decode(real)
            if self._pid != os.getpid():
                self._transform = ImageTransform(self.preprocessing, training=True)
                self._pid = os.getpid()
            fake_pixels = self._transform(fake_image)
            real_pixels = self._transform(real_image)
            return {
                "fake": {
                    "pixel_values": fake_pixels,
                    "label": 1,
                    "metadata": metadata(fake),
                },
                "real": {
                    "pixel_values": real_pixels,
                    "label": 0,
                    "metadata": metadata(real),
                },
                "pair": {
                    "fake_id": fake.id,
                    "real_id": real.id,
                    "rng_before": before,
                    "rng_after": rng_state(),
                },
            }
        except Exception as error:
            raise UCFDataError(
                {
                    "requested_fake": metadata(fake),
                    "sampled_real": metadata(real) if real is not None else None,
                    "error": f"{type(error).__name__}: {error}",
                    "rng_before": before,
                    "rng_after": rng_state(),
                }
            ) from error


class UCFCollator:
    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError("UCF cannot silently omit either half of a pair")
        ordered = [sample[role] for role in ("real", "fake") for sample in samples]
        return {
            "pixel_values": torch.stack([row["pixel_values"] for row in ordered]),
            "labels": torch.tensor([row["label"] for row in ordered], dtype=torch.long),
            "specific_labels": torch.tensor(
                [row["metadata"]["specific_label"] for row in ordered], dtype=torch.long
            ),
            "metadata": [row["metadata"] for row in ordered],
            "pairs": [sample["pair"] for sample in samples],
        }
