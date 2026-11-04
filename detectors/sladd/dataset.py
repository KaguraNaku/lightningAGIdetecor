"""Native paired decode/augmentation over explicitly bound remake manifests."""

import json
import os
import random

import cv2
import numpy as np
import torch

from detectors.fwa.generation import rng_state
from .protocol import SLADDSourcePlan
from .transforms import AUGMENTATION, SLADDPairTransform


class SLADDDataError(RuntimeError):
    marker = "SLADD_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


def metadata(row):
    return {
        "id": row.id,
        "dataset": row.dataset,
        "split": row.split,
        "video_id": row.video_id,
        "frame_index": row.frame_index,
        "category": row.category,
        "attributes": row.metadata,
    }


class SLADDPairDataset:
    def __init__(self, base, *, profile):
        self.plan = SLADDSourcePlan(base, profile=profile)
        self.root, self.header, self.manifest_path = (
            self.plan.root,
            self.plan.header,
            self.plan.manifest_path,
        )
        self.records, self.real_records = self.plan.records, self.plan.real_records
        self._pid, self._transform, self._manifest_stamp = None, None, None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {
            **self.__dict__,
            "_pid": None,
            "_transform": None,
            "_manifest_stamp": None,
        }

    def _controls(self):
        info = self.manifest_path.stat()
        stamp = info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        if self._pid != os.getpid() or self._manifest_stamp != stamp:
            self.plan.verify_manifest()
            self._manifest_stamp = stamp
        if self._pid != os.getpid():
            cv2.setNumThreads(0)
            self._transform = SLADDPairTransform()
            self._pid = os.getpid()

    def _image(self, row):
        path = self.plan.file(row.id, "image")
        if self.plan.profile == "source_filesystem_gray":
            image = cv2.imread(str(path))
        else:
            image = cv2.imdecode(
                np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
        if image is None:
            raise ValueError(f"SLADD failed to decode image:{row.id}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    def _mask(self, row):
        path = self.plan.file(row.id, "mask")
        if self.plan.profile == "source_filesystem_gray":
            if path is None:
                return np.zeros((256, 256, 1))
            mask = cv2.imread(str(path), 0)
            if mask is None:
                mask = np.zeros((256, 256))
        else:
            mask = (
                np.zeros((256, 256, 3))
                if path is None
                else cv2.imdecode(
                    np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
                )
            )
        mask = cv2.resize(mask, (256, 256)) / 255
        return np.float32(np.expand_dims(mask, axis=2))

    def _landmarks(self, row):
        path = self.plan.file(row.id, "landmarks")
        if self.plan.profile == "source_filesystem_gray":
            return (
                np.zeros((81, 2)) if path is None else np.load(path, allow_pickle=False)
            )
        binary = None if path is None else path.read_bytes()
        landmarks = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
        return np.float32(landmarks) * (256 / 256)

    def selection_receipt(self):
        return {
            **self.plan.receipt(),
            "augmentation": AUGMENTATION,
            "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
            "missing_policy": "Pin present/absent sidecars;reproduce native filesystem zero-fill/LMDB mask zero-fill and missing LMDB-landmark error;later appearance/disappearance/hash changes are fatal",
            "failure_policy": "One requested-source attempt;no class/source substitution or dropped pair;receipt retains selected fake/real and RNG",
            "worker_policy": "Per-process Albu construction;spawn-pickle contains no live transform/module/library/file handle. Sharedloader seeds workers;not original8-worker scheduling/globalRNG reproduction",
        }

    def __getitem__(self, index):
        fake, real, before, operation = (
            self.records[index],
            None,
            rng_state(),
            "real_selection",
        )
        try:
            real = self.real_records[random.randint(0, len(self.real_records) - 1)]
            operation = "control_verification"
            self._controls()
            operation = "image_decode"
            fake_image, real_image = self._image(fake), self._image(real)
            operation = "mask_decode"
            fake_mask, real_mask = self._mask(fake), self._mask(real)
            operation = "landmark_decode"
            fake_landmarks, real_landmarks = (
                self._landmarks(fake),
                self._landmarks(real),
            )
            operation = "augmentation"
            fake_values = self._transform(fake_image, fake_landmarks, fake_mask)
            real_values = self._transform(real_image, real_landmarks, real_mask)
            result = {}
            for role, row, values in (
                ("fake", fake, fake_values),
                ("real", real, real_values),
            ):
                result[role] = {
                    "pixel_values": values[0],
                    "landmarks": values[1],
                    "mask": values[2],
                    "label": row.label,
                    "specific_label": self.plan.specific_labels[row.id],
                    "metadata": metadata(row),
                }
            result["pair"] = {
                "indexed_fake_id": fake.id,
                "real_id": real.id,
                "profile": self.plan.profile,
                "rng_before": before,
                "rng_after": rng_state(),
                "bindings": {
                    "fake": self.plan.bindings[fake.id],
                    "real": self.plan.bindings[real.id],
                },
            }
            return result
        except Exception as error:
            raise SLADDDataError(
                {
                    "phase": operation,
                    "requested_fake": metadata(fake),
                    "selected_real": metadata(real) if real is not None else None,
                    "profile": self.plan.profile,
                    "attempts": 1,
                    "substitution_performed": False,
                    "rng_before": before,
                    "rng_after": rng_state(),
                    "error": f"{type(error).__name__}: {error}",
                }
            ) from error


class SLADDPairCollator:
    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError("SLADD cannot drop either member of a pair")
        ordered = [
            (role, index, sample[role])
            for role in ("real", "fake")
            for index, sample in enumerate(samples)
        ]
        return {
            "pixel_values": torch.stack([row["pixel_values"] for _, _, row in ordered]),
            "labels": torch.tensor(
                [row["label"] for _, _, row in ordered], dtype=torch.long
            ),
            "specific_labels": torch.tensor(
                [row["specific_label"] for _, _, row in ordered], dtype=torch.long
            ),
            "landmarks": torch.stack([row["landmarks"] for _, _, row in ordered]),
            "masks": torch.stack([row["mask"] for _, _, row in ordered]),
            "metadata": [
                {
                    **row["metadata"],
                    "pair_index": index,
                    "view": role,
                    "pair_id": samples[index]["pair"]["indexed_fake_id"],
                }
                for role, index, row in ordered
            ],
            "pairs": [sample["pair"] for sample in samples],
        }
