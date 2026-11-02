"""Per-source generation retains native neighbor/augmentation RNG sequencing."""

import hashlib
import json
import os

import cv2
import numpy as np

from detectors.fwa.generation import rng_state
from .bank import FaceBank
from .generation import FaceXrayGenerator, assemble_pairs


class FaceXrayGenerationError(RuntimeError):
    marker = "FACEXRAY_GENERATION_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


def rng_hash(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


class FaceXrayDataset:
    def __init__(self, dataset, bank):
        self.base = dataset
        self.header, self.records = dataset.header, dataset.records
        self.manifest_path, self.root = dataset.manifest_path, dataset.root
        self.bank = FaceBank(bank, dataset)
        self._pid, self._generator = None, None

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_generator": None}

    def __len__(self):
        return len(self.records)

    def _decode(self, index):
        sample = self.base[index]
        record = self.records[index]
        if sample is None:
            raise ValueError(f"FaceXray source could not decode: {record.id}")
        if sample["metadata"]["id"] != record.id or sample["label"] != record.label:
            raise ValueError(
                "FaceXray decoded source differs from bank/manifest binding"
            )
        image = cv2.resize(
            np.asarray(sample["image"].convert("RGB")),
            (256, 256),
            interpolation=cv2.INTER_CUBIC,
        )
        return sample, image

    def selection_receipt(self):
        return {
            **self.bank.receipt(),
            "parent_count": len(self),
            "real_backgrounds": sum(row.label == 0 for row in self.records),
            "fake_backgrounds": sum(row.label == 1 for row in self.records),
            "source_filter": "none; all explicit manifest background rows retained",
            "foreground_scope": "Every ordered neighbor reference belongs to this exact train inventory; no validation/external hidden source",
        }

    def __getitem__(self, index):
        record = self.records[index]
        parent = {
            "id": record.id,
            "dataset": record.dataset,
            "split": record.split,
            "video_id": record.video_id,
            "frame_index": record.frame_index,
            "category": record.category,
            "attributes": record.metadata,
        }
        before, selected = rng_state(), None
        try:
            if self._pid != os.getpid():
                self._generator, self._pid = FaceXrayGenerator(), os.getpid()
            selected, self_fallback = self.bank.foreground(index)
            foreground, foreground_pixels = self._decode(selected)
            background = {}

            def read_background():
                sample, image = self._decode(index)
                background["sample"] = sample
                return image

            fake, real, mask, generated = self._generator.generate(
                foreground_pixels,
                self.bank.landmarks(selected),
                read_background,
                self.bank.landmarks(index),
            )
            if background["sample"]["metadata"] != parent:
                raise ValueError(
                    "FaceXray background identity changed during generation"
                )
            after = rng_state()
            return {
                "fake": fake,
                "real": real,
                "mask": mask,
                "label": record.label,
                "metadata": parent,
                "rng_before": before,
                "rng_after": after,
                "generation": {
                    **generated,
                    "source_label": record.label,
                    "generated_label": 1,
                    "fallback": False,
                    "neighbor_fallback_to_self": self_fallback,
                    "foreground": foreground["metadata"],
                    "foreground_label": foreground["label"],
                    "bank_sha256": self.bank.identity["sha256"],
                    "rng_before_sha256": rng_hash(before),
                    "rng_after_sha256": rng_hash(after),
                },
            }
        except Exception as error:
            raise FaceXrayGenerationError(
                {
                    "background": parent,
                    "foreground_id": self.records[selected].id
                    if selected is not None
                    else None,
                    "bank": self.bank.identity,
                    "error": f"{type(error).__name__}: {error}",
                    "rng_before": before,
                    "rng_after": rng_state(),
                }
            ) from error


class FaceXrayPairCollator:
    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError(
                "FaceXray generation cannot silently skip an invalid parent"
            )
        before_shuffle = rng_state()
        batch = assemble_pairs(
            [
                (sample["fake"], sample["real"], sample["mask"], sample["label"])
                for sample in samples
            ]
        )
        order = batch.pop("view_order")
        batch["metadata"] = [
            {**samples[index]["metadata"], "view": view} for index, view in order
        ]
        batch["sources"] = [
            {
                "metadata": sample["metadata"],
                "label": sample["label"],
                "generation": sample["generation"],
            }
            for sample in samples
        ]
        batch["generation_rng"] = {
            "before": samples[0]["rng_before"],
            "before_shuffle": before_shuffle,
            "after_generation": samples[-1]["rng_after"],
            "after": rng_state(),
        }
        return batch
