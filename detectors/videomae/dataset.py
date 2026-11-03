"""T16/first32 clip data preserving native landmark IO and augmented val."""

import hashlib
import json
import os
import random

from detectors.i3d.dataset import I3DDataset, I3DCollator
from .landmarks import bind, load


class VideoMAEDataError(RuntimeError):
    marker = "VIDEOMAE_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(self.marker + json.dumps(receipt))


class VideoMAEDataset(I3DDataset):
    def __init__(self, base, *, stage, seed):
        # Reuse hash-bound complete-frame inventory, native T16 selection and
        # source-split roles. Do not reuse I3D's landmark-free pixel transform.
        super().__init__(base, stage=stage, seed=seed)
        self.landmark_bindings = {record.id: bind(self.root, record) for record in self.source_records}
        identity = hashlib.sha256()
        present = 0
        for name, descriptor in self.landmark_bindings.items():
            identity.update(json.dumps([name, descriptor], sort_keys=True).encode())
            identity.update(b"\n")
            present += descriptor["sha256"] is not None
        self.protocol.update({
            "method": "videomae",
            "augmentation": "Native Albu1.3.1 keypoint-conditioned shared clip seed; train AND validation augmented; evaluation unaugmented",
            "validation": "Native augmented val pixels, deterministic val clip selection; conditional seed parity, not native global process RNG",
            "landmarks": {
                "policy": "Native absolute-image-string replacement, filesystem NPY unscaled; absent81x2 float64 zeros",
                "binding": "Captured at dataset construction over entire declared source pool, including excluded tails/short videos; not author corpus discovery",
                "identity_sha256": identity.hexdigest(), "present": present,
                "missing": len(self.landmark_bindings) - present,
                "bindings": self.landmark_bindings,
                "output": "Original untransformed landmark tensors, as native __getitem__; ignored by Detector",
            },
            "decode_failure": "Fatal one-attempt receipt; preserve requested clip/video/label instead of native first-clip recursion",
        })

    def __getitem__(self, index):
        import cv2
        import torch
        from .transforms import VideoMAEClipTransform

        clip = self.records[index]
        if self._pid != os.getpid():
            cv2.setNumThreads(0)
            self._pid = os.getpid()
            self._transform = VideoMAEClipTransform(augment=self.stage in {"train", "validation"})
        augmentation_seed = random.randint(0, 2**32 - 1)
        tensors, landmarks, annotations = [], [], []
        for position, frame in enumerate(clip.frames):
            operation = "image_decode"
            try:
                pixels = cv2.imread(str(self.root / frame.path))
                if pixels is None:
                    raise ValueError("OpenCV could not decode image")
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                pixels = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_CUBIC)
                operation = "landmark_decode"
                points, annotation = load(self.root, self.landmark_bindings[frame.id])
                operation = "augmentation_and_tensor"
                image, points_tensor = self._transform(pixels, points, augmentation_seed=augmentation_seed)
                tensors.append(image)
                landmarks.append(points_tensor)
                annotations.append(annotation)
            except Exception as error:
                raise VideoMAEDataError({
                    "requested_clip": clip.id, "dataset": clip.dataset, "split": clip.split,
                    "video_id": clip.video_id, "label": clip.label, "failed_source_id": frame.id,
                    "path": frame.path, "landmark": self.landmark_bindings[frame.id],
                    "temporal_position": position, "operation": operation,
                    "attempts": 1, "substitution_performed": False,
                    "error": f"{type(error).__name__}: {error}",
                }) from error
        return {
            "pixel_values": torch.stack(tensors), "landmarks": torch.stack(landmarks),
            "label": clip.label,
            "metadata": {
                "id": clip.id, "dataset": clip.dataset, "split": clip.split,
                "video_id": clip.video_id, "prediction_unit": "clip",
                "source_frame_ids": [frame.id for frame in clip.frames],
                "source_pool_frame_ids": list(clip.source_frame_ids),
                "physical_frame_indices": list(clip.frame_positions),
                "attributes": {"augmentation_seed": augmentation_seed, "landmarks": annotations,
                               "augmented": self.stage in {"train", "validation"}},
            },
        }


class VideoMAECollator(I3DCollator):
    def __call__(self, samples):
        import torch

        batch = super().__call__(samples)
        batch["landmarks"] = torch.stack([sample["landmarks"] for sample in samples])
        return batch
