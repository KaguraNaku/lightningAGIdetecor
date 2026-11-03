"""Explicit T4 adapters for the formal entry and specialized local TALL body."""

import json
import os
import random

from detectors.i3d.inventory import I3DInventory
from .clips import select_clips
from .transforms import PROFILES


class TALLDataError(RuntimeError):
    marker = "TALL_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(self.marker + json.dumps(receipt))


class TALLDataset:
    def __init__(self, base, *, stage, seed, profile):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown TALL dataset stage")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("TALL selection seed must be unsigned32-bit integer")
        if not isinstance(profile, str) or profile not in PROFILES:
            raise ValueError(
                "Select native_entrypoint or specialized_cutout explicitly"
            )
        inventory = I3DInventory(base.manifest_path, dataset=base)
        allowed = {
            "train": {"train"},
            "validation": {"val"},
            "evaluation": {"val", "test"},
        }[stage]
        if inventory.header["split"] not in allowed:
            raise ValueError(f"TALL {stage} requires source split in {sorted(allowed)}")
        self.base, self.root = base, base.root
        self.header, self.manifest_path = inventory.header, inventory.manifest_path
        self.source_records = inventory.source_records
        self.stage, self.seed, self.profile = stage, seed, profile
        rng = random.Random(seed)
        self.records, selection = select_clips(
            inventory.pools, training=stage == "train", rng=rng
        )
        if not self.records:
            raise ValueError("No complete4-frame clips in TALL source snapshot")
        self.augment = stage in PROFILES[profile]["augmentation_stages"]
        self.cutout = PROFILES[profile]["cutout"]
        self.protocol = {
            "source": inventory.provenance,
            "selection": selection,
            "selection_seed": seed,
            "selection_python_rng_after": rng.getstate(),
            "data_profile": profile,
            "reference": PROFILES[profile]["reference"],
            "augmentation_stages": list(PROFILES[profile]["augmentation_stages"]),
            "cutout_in_all_stages": self.cutout,
            "cutout_draws": "Two global NumPy randint calls after worker-state-preserving frame transforms",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "worker_RNG": "Restore Python/NumPy state instead of native entropy reseeding;not native full process RNG replay",
            "decode_failure": "Fatal requested-identity receipt;no recursive sample0 substitution",
            "specialized_collator_repair": self.cutout,
            "specialized_collator_note": "Original specialized getitem returns4 fields but inherited collator expects5;explicit named fields preserve images/labels",
            "full_experiment_reproduction": False,
        }
        self._pid, self._transform = None, None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_transform": None}

    def __getitem__(self, index):
        import cv2
        import numpy as np
        import torch
        from .transforms import TALLClipTransform, apply_cutout

        clip = self.records[index]
        if self._pid != os.getpid():
            cv2.setNumThreads(0)
            self._transform = TALLClipTransform(augment=self.augment)
            self._pid = os.getpid()
        augmentation_seed = random.randint(0, 2**32 - 1)
        frames = []
        for position, frame in enumerate(clip.frames):
            operation = "image_decode"
            try:
                pixels = cv2.imread(str(self.root / frame.path))
                if pixels is None:
                    raise ValueError("OpenCV could not decode image")
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                pixels = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_CUBIC)
                operation = "augmentation_and_ImageNet_normalization"
                frames.append(
                    self._transform(pixels, augmentation_seed=augmentation_seed)
                )
            except Exception as error:
                raise TALLDataError(
                    {
                        "requested_clip": clip.id,
                        "dataset": clip.dataset,
                        "split": clip.split,
                        "video_id": clip.video_id,
                        "label": clip.label,
                        "failed_source_id": frame.id,
                        "path": frame.path,
                        "temporal_position": position,
                        "physical_frame_index": frame.frame_index,
                        "operation": operation,
                        "attempts": 1,
                        "substitution_performed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                ) from error
        pixels = torch.stack(frames)
        cutout = apply_cutout(pixels, rng=np.random) if self.cutout else None
        return {
            "pixel_values": pixels,
            "label": clip.label,
            "metadata": {
                "id": clip.id,
                "dataset": clip.dataset,
                "split": clip.split,
                "video_id": clip.video_id,
                "prediction_unit": "clip",
                "source_frame_ids": [f.id for f in clip.frames],
                "source_pool_frame_ids": list(clip.source_frame_ids),
                "physical_frame_indices": list(clip.frame_positions),
                "category": clip.frames[0].category,
                "attributes": {
                    "data_profile": self.profile,
                    "augmentation_seed": augmentation_seed,
                    "augmented": self.augment,
                    "cutout": cutout,
                },
            },
        }


class TALLCollator:
    def __init__(self, *, profile):
        if profile not in PROFILES:
            raise ValueError("Unknown TALL data recipe")
        self.profile = profile

    def __call__(self, samples):
        import torch

        if not samples or any(sample is None for sample in samples):
            raise ValueError("TALL collation cannot silently omit failed clips")
        for sample in samples:
            pixels = sample["pixel_values"]
            if (
                not isinstance(pixels, torch.Tensor)
                or pixels.shape != (4, 3, 224, 224)
                or pixels.dtype != torch.float32
            ):
                raise ValueError("TALL collator requires FP32 T4/RGB224 clips")
            if type(sample["label"]) is not int or sample["label"] not in {0, 1}:
                raise ValueError("TALL collator requires binary integer labels")
            meta = sample["metadata"]
            if (
                meta.get("prediction_unit") != "clip"
                or meta.get("attributes", {}).get("data_profile") != self.profile
            ):
                raise ValueError(
                    "TALL collation cannot mix prediction units or data recipes"
                )
        return {
            "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
            "labels": torch.tensor([s["label"] for s in samples], dtype=torch.int64),
            "metadata": [s["metadata"] for s in samples],
        }
