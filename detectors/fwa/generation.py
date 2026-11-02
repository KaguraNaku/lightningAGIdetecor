"""Method-owned native FWA augmentation and doubled generated/original batches."""

from dataclasses import dataclass
import hashlib
import json
import random

import cv2
import numpy as np
from PIL import Image
import torch

from .geometry import aligned_face, aligned_points, warp_back, LazyDlibAlignment
from .masks import face_mask


def rng_state():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
    }


def rng_digest():
    return hashlib.sha256(json.dumps(rng_state()).encode()).hexdigest()


def pixel_digest(pixels):
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(pixels)).cast("B")
    ).hexdigest()


def augmentation_library():
    import albumentations as A

    if A.__version__ != "1.3.1":
        raise RuntimeError("Native FWA generation requires albumentations==1.3.1")
    return A


def downscale_transform():
    A = augmentation_library()

    class RandomDownScale(A.core.transforms_interface.ImageOnlyTransform):
        def apply(self, img, **params):
            height, width, _ = img.shape
            ratio = [2, 4][np.random.randint(2)]
            down = cv2.resize(
                img,
                (int(width / ratio), int(height / ratio)),
                interpolation=cv2.INTER_NEAREST,
            )
            return cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)

    return RandomDownScale(p=1)


class FWAGenerator:
    def __init__(self, aligner):
        self.aligner = aligner

    @staticmethod
    def blended_aug(image):
        A = augmentation_library()
        transform = A.Compose(
            [
                A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3),
                    sat_shift_limit=(-0.3, 0.3),
                    val_shift_limit=(-0.3, 0.3),
                    p=0.3,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3
                ),
                A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            ]
        )
        return transform(image=image)["image"]

    @staticmethod
    def original_aug(image):
        A = augmentation_library()
        transform = A.Compose(
            [
                A.Compose(
                    [
                        A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                        A.HueSaturationValue(
                            hue_shift_limit=(-0.3, 0.3),
                            sat_shift_limit=(-0.3, 0.3),
                            val_shift_limit=(-0.3, 0.3),
                            p=1,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.1),
                            contrast_limit=(-0.1, 0.1),
                            p=1,
                        ),
                    ],
                    p=1,
                ),
                A.OneOf(
                    [
                        downscale_transform(),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                    ],
                    p=1,
                ),
            ],
            p=1.0,
        )
        return transform(image=image)["image"]

    def blend(self, image):
        faces = self.aligner(image)
        receipt = {"faces_detected": len(faces), "fallback": not bool(faces)}
        aligned = [
            (aligned_face(image, matrix), aligned_points(matrix, points))
            for matrix, points in faces
        ]
        if not aligned:
            return None, None, receipt
        face, points = aligned[0]
        scale = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        scaled = cv2.resize(face, (0, 0), fx=scale, fy=scale)
        blurred = cv2.GaussianBlur(scaled, (5, 5), 0)
        resized = cv2.resize(blurred, (face.shape[1], face.shape[0]))
        mask = face_mask(points.astype(np.float32), resized, std=20, deform=True)
        masked = cv2.bitwise_and(resized, resized, mask=mask)
        augmented = np.array(self.blended_aug(image))
        generated, generated_mask = warp_back(augmented, masked, faces[0][0])
        receipt.update(
            scale=scale,
            matrix=faces[0][0].tolist(),
            landmarks=faces[0][1].tolist(),
            mask_sha256=pixel_digest(mask),
        )
        return generated, generated_mask, receipt

    def post_process(self, image):
        swapped = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Source discards this result; its RNG consumption remains required.
        self.blended_aug(swapped)
        values = np.array(Image.fromarray(np.uint8(swapped)), copy=True)
        tensor = torch.from_numpy(values).permute(2, 0, 1).contiguous().float().div(255)
        normalization = tensor.new_tensor([0.5] * 3)[:, None, None]
        return tensor.sub(normalization).div(normalization)

    def pair(self, image, source_label):
        before = rng_digest()
        rgb = cv2.resize(
            np.asarray(image.convert("RGB")), (256, 256), interpolation=cv2.INTER_CUBIC
        )
        generated, mask, receipt = self.blend(rgb.copy())
        original = np.array(self.original_aug(rgb.copy()))
        generated_label = 1
        if generated is None or mask is None:
            generated, generated_label = original.copy(), source_label
        generated_tensor = self.post_process(generated)
        original_tensor = self.post_process(original)
        receipt.update(
            rng_before=before,
            rng_after=rng_digest(),
            source_label=source_label,
            generated_label=generated_label,
            generated_sha256=pixel_digest(generated_tensor.numpy()),
            original_sha256=pixel_digest(original_tensor.numpy()),
        )
        return (
            (generated_tensor, generated_label),
            (original_tensor, source_label),
            receipt,
        )


@dataclass
class FWAPairCollator:
    predictor: dict
    generator: object = None

    def __post_init__(self):
        if self.generator is None:
            if self.predictor is None:
                raise ValueError(
                    "FWA training requires an explicit landmark predictor path"
                )
            self.generator = FWAGenerator(LazyDlibAlignment(self.predictor))

    def __call__(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        before = rng_state()
        generated, originals, sources = [], [], []
        for sample in samples:
            fake, real, receipt = self.generator.pair(sample["image"], sample["label"])
            parent = sample["metadata"]
            generated.append((fake[0], fake[1], {**parent, "view": "generated"}))
            originals.append((real[0], real[1], {**parent, "view": "original"}))
            sources.append(
                {"metadata": parent, "label": sample["label"], "generation": receipt}
            )
        combined = generated + originals
        random.shuffle(combined)
        pixels, labels, metadata = zip(*combined, strict=True)
        return {
            "pixel_values": torch.stack(pixels),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": list(metadata),
            "sources": sources,
            "generation_rng": {"before": before, "after": rng_state()},
        }
