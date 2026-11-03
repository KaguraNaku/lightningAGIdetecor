"""T8 clip composition over the existing immutable all-frame manifest contract."""

import json
import os
import random

from detectors.i3d.inventory import I3DInventory
from .clips import select_clips


class STILDataError(RuntimeError):
    marker = "STIL_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(self.marker + json.dumps(receipt))


class STILDataset:
    def __init__(self, base, *, stage, seed):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown STIL dataset stage")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("STIL selection seed must be unsigned32-bit integer")
        # Reuse only the method-independent all-frame binding/validation,not
        # I3D's fixedT16 selection,dataset,or validation augmentation policy.
        inventory = I3DInventory(base.manifest_path, dataset=base)
        required = {
            "train": {"train"},
            "validation": {"val"},
            "evaluation": {"val", "test"},
        }[stage]
        if inventory.header["split"] not in required:
            raise ValueError(
                f"STIL {stage} requires source split in {sorted(required)}"
            )
        self.base, self.root = base, base.root
        self.header, self.manifest_path = inventory.header, inventory.manifest_path
        self.source_records = inventory.source_records
        self.stage, self.seed = stage, seed
        rng = random.Random(seed)
        self.records, selection = select_clips(
            inventory.pools, training=stage == "train", rng=rng
        )
        if not self.records:
            raise ValueError("No complete8-frame clips in STIL source snapshot")
        self.protocol = {
            "source": inventory.provenance,
            "selection": selection,
            "selection_seed": seed,
            "selection_python_rng_after": rng.getstate(),
            "selection_time": "Once at construction,before workers",
            "augmentation": "Native Albu1.3.1 shared clip seed;train and validation augmented,evaluation unaugmented;no masks/landmarks",
            "worker_RNG": "Restore Python/NumPy state after each frame instead of native entropy reseeding",
            "full_native_process_RNG_reproduced": False,
            "decode_failure": "One attempted decode,fatal identity receipt;no recursive sample0/label substitution",
        }
        self._pid, self._transform = None, None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_transform": None}

    def __getitem__(self, index):
        import cv2
        import torch
        from .transforms import STILClipTransform

        clip = self.records[index]
        if self._pid != os.getpid():
            cv2.setNumThreads(0)
            self._transform = STILClipTransform(
                augment=self.stage in {"train", "validation"}
            )
            self._pid = os.getpid()
        augmentation_seed = random.randint(0, 2**32 - 1)
        tensors = []
        for position, frame in enumerate(clip.frames):
            operation = "image_decode"
            try:
                pixels = cv2.imread(str(self.root / frame.path))
                if pixels is None:
                    raise ValueError("OpenCV could not decode image")
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                pixels = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_CUBIC)
                operation = "augmentation_and_tensor"
                tensors.append(
                    self._transform(pixels, augmentation_seed=augmentation_seed)
                )
            except Exception as error:
                raise STILDataError(
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
                "category": clip.frames[0].category,
                "attributes": {
                    "augmentation_seed": augmentation_seed,
                    "augmented": self.stage in {"train", "validation"},
                },
            },
        }


class STILCollator:
    def __call__(self, samples):
        import torch

        if not samples or any(sample is None for sample in samples):
            raise ValueError("STIL batches cannot silently omit failed clips")
        for sample in samples:
            pixels = sample["pixel_values"]
            if (
                not isinstance(pixels, torch.Tensor)
                or tuple(pixels.shape) != (8, 3, 224, 224)
                or pixels.dtype != torch.float32
            ):
                raise ValueError("STIL collator requires FP32 RGB224/T8 clips")
            if type(sample["label"]) is not int or sample["label"] not in {0, 1}:
                raise ValueError("STIL collator requires binary integer labels")
            if sample["metadata"].get("prediction_unit") != "clip":
                raise ValueError("STIL metadata must declare clip prediction unit")
        return {
            "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.int64
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }
