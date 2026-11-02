"""Clip dataset composition; decode failures are fatal without relabeling."""

import json
import os
import random

from .clips import select_clips
from .inventory import I3DInventory


class I3DDataError(RuntimeError):
    marker = "I3D_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


class I3DDataset:
    def __init__(self, base, *, stage, seed):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown I3D dataset stage")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("I3D selection seed must be an unsigned32-bit integer")
        inventory = I3DInventory(base.manifest_path)
        expected_split = {"train": "train", "validation": "val"}.get(stage)
        if expected_split is not None and inventory.header["split"] != expected_split:
            raise ValueError(
                f"I3D {stage} adaptation requires a {expected_split} source split"
            )
        if (
            inventory.header != base.header
            or tuple(base.records) != inventory.source_records
        ):
            raise ValueError("I3D base dataset differs from its bound source snapshot")
        self.base, self.root = base, base.root
        self.header, self.manifest_path = inventory.header, inventory.manifest_path
        self.source_records = inventory.source_records
        self.stage, self.seed = stage, seed
        rng = random.Random(seed)
        self.records, selection = select_clips(
            inventory.pools, training=stage == "train", rng=rng
        )
        if not self.records:
            raise ValueError("No complete16-frame clips in the I3D input snapshot")
        self.protocol = {
            "source": inventory.provenance,
            "selection": selection,
            "selection_seed": seed,
            "selection_python_rng_after": rng.getstate(),
            "selection_time": "Once at dataset construction, before workers",
            "augmentation": "Native shared-seed pixel recipe; restore worker RNG after each frame instead of native entropy reset",
            "validation": "Deterministic evaluation pixel profile, source-domain selection; native augmented val mode not reproduced",
            "decode_failure": "One attempted decode per selected frame, fatal receipt; never substitute another clip/video/label",
            "full_native_process_RNG_reproduced": False,
        }
        self._pid, self._transform = None, None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_transform": None}

    def __getitem__(self, index):
        # Heavy image libraries are loaded only when actual clip IO begins.
        import cv2
        import torch
        from .transforms import I3DClipTransform

        clip = self.records[index]
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._transform = I3DClipTransform(training=self.stage == "train")
        augmentation_seed = random.randint(0, 2**32 - 1)
        tensors = []
        for temporal_position, frame in enumerate(clip.frames):
            path = self.root / frame.path
            try:
                pixels = cv2.imread(str(path))
                if pixels is None:
                    raise ValueError("OpenCV could not decode image")
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                pixels = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_CUBIC)
            except Exception as error:
                raise I3DDataError(
                    {
                        "requested_clip": clip.id,
                        "dataset": clip.dataset,
                        "split": clip.split,
                        "video_id": clip.video_id,
                        "label": clip.label,
                        "failed_source_id": frame.id,
                        "path": frame.path,
                        "temporal_position": temporal_position,
                        "physical_frame_index": frame.frame_index,
                        "attempts": 1,
                        "substitution_performed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                ) from error
            tensors.append(self._transform(pixels, augmentation_seed=augmentation_seed))
        return {
            "pixel_values": torch.stack(tensors),
            "label": clip.label,
            "metadata": {
                "id": clip.id,
                "dataset": clip.dataset,
                "split": clip.split,
                "video_id": clip.video_id,
                "prediction_unit": "clip",
                "source_frame_ids": [frame.id for frame in clip.frames],
                "source_pool_frame_ids": list(clip.source_frame_ids),
                "physical_frame_indices": list(clip.frame_positions),
                "attributes": {"augmentation_seed": augmentation_seed},
            },
        }


class I3DCollator:
    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError("I3D batches cannot silently omit failed clips")
        import torch

        return {
            "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }
