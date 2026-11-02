"""Explicit five-domain inventories and bounded, same-video neighboring retries."""

from dataclasses import replace
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
import torch

from data.manifest import sha256
from detectors.fwa.generation import rng_state
from .transforms import LSDATransform


DEFAULT_LABEL_MAP = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 2, "FF-FS": 3, "FF-NT": 4}


class LSDADataError(RuntimeError):
    marker = "LSDA_DATA_FAILURE:"

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


class LSDADataset:
    def __init__(self, dataset, *, label_map=None, max_decode_attempts=32):
        if dataset.header["split"] != "train":
            raise ValueError("LSDA grouped sampling is a train-only adapter")
        if type(max_decode_attempts) is not int or max_decode_attempts < 1:
            raise ValueError("LSDA max_decode_attempts must be a positive integer")
        self.base, self.root = dataset, dataset.root
        self.header, self.manifest_path = dataset.header, dataset.manifest_path
        self.label_map = dict(DEFAULT_LABEL_MAP if label_map is None else label_map)
        if any(
            not isinstance(key, str) or type(value) is not int or not 0 <= value < 5
            for key, value in self.label_map.items()
        ):
            raise ValueError("LSDA requires an explicit five-domain category map")
        self.max_decode_attempts = max_decode_attempts
        slots, videos, ids, self.source_positions = {}, {}, set(), {}
        for position, row in enumerate(dataset.records):
            group = row.metadata.get("lsda_group")
            slot = row.metadata.get("lsda_frame_slot")
            declared = self.label_map.get(row.category)
            domain = row.metadata.get("specific_label", declared)
            if (
                type(group) is not int
                or group < 0
                or type(slot) is not int
                or not 0 <= slot < 32
                or type(domain) is not int
                or not 0 <= domain < 5
                or (domain == 0) != (row.label == 0)
                or (declared is not None and domain != declared)
                or not row.video_id
                or row.split != "train"
                or row.dataset != dataset.header["dataset"]
            ):
                raise ValueError(
                    f"LSDA {row.id} requires train-source identity, explicit lsda_group/lsda_frame_slot0..31 and consistent specific_label0..4"
                )
            key, video_key = (group, domain, slot), (group, domain)
            if key in slots or row.id in ids:
                raise ValueError(
                    f"LSDA duplicate group/domain/slot binding or id:{row.id}"
                )
            identity = row.dataset, row.category, row.video_id
            if videos.setdefault(video_key, identity) != identity:
                raise ValueError("LSDA neighboring slots must belong to the same video")
            slots[key] = replace(
                row, metadata={**row.metadata, "specific_label": domain}
            )
            ids.add(row.id)
            self.source_positions[row.id] = position
        groups = sorted({key[0] for key in slots})
        if not groups or groups != list(range(len(groups))):
            raise ValueError(
                "LSDA group ids must form a nonempty contiguous0-based order"
            )
        self.num_groups = len(groups)
        expected = {
            (group, domain, slot)
            for group in groups
            for domain in range(5)
            for slot in range(32)
        }
        if slots.keys() != expected:
            raise ValueError(
                "LSDA requires all32 slots for all five domains in every group"
            )
        self.records = [slots[key] for key in sorted(slots)]
        self._pid, self._transform = None, None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_transform": None}

    def _decode(self, row):
        if row.storage == "image":
            path = self.root / row.path
            if not path.exists():
                raise FileNotFoundError(path)
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise ValueError(f"LSDA failed to decode {row.id}")
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        else:
            pixels = np.asarray(self.base._parquet_image(row).convert("RGB"))
        return Image.fromarray(
            cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_CUBIC)
        )

    def selection_receipt(self):
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256(self.manifest_path),
            "policy": "Explicit group/domain/slot binding, canonical group->real/DF/F2F/FS/NT->32slot order; sampler selects one slot per group per epoch, not every inventory row",
            "inventory_rows": len(self.records),
            "groups": self.num_groups,
            "sampled_images_per_epoch": self.num_groups * 5,
            "label_map": self.label_map,
            "max_decode_attempts": self.max_decode_attempts,
            "sources": [
                {
                    **metadata(row),
                    "dataset_index": index,
                    "manifest_position": self.source_positions[row.id],
                }
                for index, row in enumerate(self.records)
            ],
            "repairs": [
                "Explicit roots replace eager CUDA-name/hardcoded drive discovery",
                "Manifest sources replace native train/val/test and c23/c40 fallback discovery",
                "Persistent native decode recursion becomes bounded same-video attempts with a fatal receipt",
            ],
        }

    def __getitem__(self, index):
        requested = self.records[index]
        before, attempts = rng_state(), []
        for attempt in range(self.max_decode_attempts):
            actual = self.records[index]
            try:
                image = self._decode(actual)
            except Exception as error:
                attempts.append(
                    {
                        "source": metadata(actual),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if attempt + 1 == self.max_decode_attempts:
                    raise LSDADataError(
                        {
                            "requested": metadata(requested),
                            "attempts": attempts,
                            "max_decode_attempts": self.max_decode_attempts,
                            "rng_before": before,
                            "rng_after": rng_state(),
                        }
                    ) from error
                slot = actual.metadata["lsda_frame_slot"]
                index += (
                    1 if slot == 0 else -1 if slot == 31 else random.choice([-1, 1])
                )
                continue
            attempts.append({"source": metadata(actual), "error": None})
            if self._pid != os.getpid():
                self._transform, self._pid = LSDATransform(), os.getpid()
            pixels = self._transform(image)
            return {
                "pixel_values": pixels,
                "label": actual.label,
                "specific_label": actual.metadata["specific_label"],
                "metadata": metadata(actual),
                "receipt": {
                    "requested": metadata(requested),
                    "actual": metadata(actual),
                    "attempts": attempts,
                    "rng_before": before,
                    "rng_after": rng_state(),
                },
            }


class LSDACollator:
    def __call__(self, samples):
        if not samples or len(samples) % 5 or any(row is None for row in samples):
            raise ValueError(
                "LSDA requires complete five-item groups without silent drops"
            )
        images = torch.stack([row["pixel_values"] for row in samples])
        labels = torch.tensor(
            [row["specific_label"] for row in samples], dtype=torch.long
        )
        images, labels = images.view(-1, 5, 3, 256, 256), labels.view(-1, 5)
        valid, rejected = [], []
        for index, domains in enumerate(labels):
            group = samples[index * 5 : (index + 1) * 5]
            if set(domains.tolist()) != {0, 1, 2, 3, 4}:
                rejected.append(
                    {
                        "batch_group": index,
                        "labels": domains.tolist(),
                        "sources": [row["metadata"] for row in group],
                        "receipts": [row["receipt"] for row in group],
                    }
                )
                continue
            if domains.tolist() != list(range(5)):
                raise ValueError(
                    "LSDA valid domain sets must retain canonical real/DF/F2F/FS/NT order"
                )
            bindings = [row["receipt"]["requested"]["attributes"] for row in group]
            if (
                len({row["lsda_group"] for row in bindings}) != 1
                or len({row["lsda_frame_slot"] for row in bindings}) != 1
            ):
                raise ValueError("LSDA must sample one shared requested slot per group")
            valid.append(index)
        if not valid:
            raise ValueError(
                "No valid groups found in this batch. " + json.dumps(rejected)
            )
        selected = [
            samples[index * 5 + domain] for index in valid for domain in range(5)
        ]
        domains = labels[valid]
        return {
            "pixel_values": images[valid],
            "labels": torch.where(domains == 0, 0, 1),
            "specific_labels": domains,
            "metadata": [row["metadata"] for row in selected],
            "receipts": [row["receipt"] for row in selected],
            "rejected_groups": rejected,
        }
