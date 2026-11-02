"""Native SBI synchronized self blending, with errors retained before update."""

import json
import hashlib
import random

import cv2
import numpy as np
import torch

from detectors.fwa.generation import (
    augmentation_library,
    pixel_digest,
    rng_digest,
    rng_state,
)
from .masks import hull_mask


class SBIGenerationError(RuntimeError):
    marker = "SBI_GENERATION_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


def blend_mask(mask):
    height, width = mask.shape
    size_h, size_w = np.random.randint(192, 257), np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    first = random.randrange(5, 26, 2)
    second = random.randrange(5, 26, 2)
    blurred = cv2.GaussianBlur(mask, (first, first), 0)
    blurred = blurred / blurred.max()
    blurred[blurred < 1] = 0
    blurred = cv2.GaussianBlur(blurred, (second, second), np.random.randint(5, 46))
    blurred = blurred / blurred.max()
    blurred = cv2.resize(blurred, (width, height))
    return blurred.reshape(blurred.shape + (1,))


class SBIGenerator:
    def __init__(self):
        from .transforms import RandomDownScale

        A = augmentation_library()
        self.transforms = A.Compose(
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
            ],
            additional_targets={"image1": "image"},
            p=1.0,
        )
        self.source_transforms = A.Compose(
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
                        RandomDownScale(p=1),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                    ],
                    p=1,
                ),
            ],
            p=1.0,
        )

    @staticmethod
    def randaffine(image, mask):
        A = augmentation_library()
        affine = A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1,
        )
        elastic = A.ElasticTransform(alpha=50, sigma=7, alpha_affine=0, p=1)
        transformed = affine(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        transformed = elastic(image=image, mask=mask)
        # Native code discards elastic's IMAGE while retaining its MASK.
        return image, transformed["mask"]

    def self_blending(self, image, landmarks):
        if np.random.rand() < 0.25:
            landmarks = landmarks[:68]
        kind = random.choice([0, 1, 2, 3])
        mask = hull_mask(landmarks, image, kind)
        source = image.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))["image"]
        else:
            image = self.source_transforms(image=image.astype(np.uint8))["image"]
        source, mask = self.randaffine(source, mask)
        mask = blend_mask(mask)
        ratio = [0.25, 0.5, 0.75, 1, 1, 1][np.random.randint(6)]
        mask *= ratio
        generated = mask * source + (1 - mask) * image
        return image.astype(np.uint8), generated.astype(np.uint8), mask

    def generate(self, image, landmarks):
        original, generated, mask = self.self_blending(image.copy(), landmarks.copy())
        result = self.transforms(
            image=generated.astype("uint8"), image1=original.astype("uint8")
        )
        return result["image"], result["image1"], mask


def normalized(image):
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        .float()
        .div(255)
    )
    return tensor.sub(tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]).div(
        tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    )


class SBIPairCollator:
    def __init__(self):
        self.generator = SBIGenerator()

    def __call__(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        before = rng_state()
        originals, generated, sources = [], [], []
        for sample in samples:
            if sample["label"] != 0:
                raise ValueError(
                    "SBI source filtering must happen before batching, not in collation"
                )
            parent = sample["metadata"]
            source_rng = rng_state()
            image = cv2.resize(
                np.asarray(sample["image"].convert("RGB")),
                (256, 256),
                interpolation=cv2.INTER_CUBIC,
            )
            try:
                fake, real, mask = self.generator.generate(image, sample["landmarks"])
            except Exception as error:
                raise SBIGenerationError(
                    {
                        "source": parent,
                        "landmark": parent["attributes"]["landmarks"],
                        "error": f"{type(error).__name__}: {error}",
                        "rng_before": source_rng,
                        "rng_after": rng_state(),
                    }
                ) from error
            real_tensor, fake_tensor = normalized(real), normalized(fake)
            originals.append((real_tensor, 0, {**parent, "view": "original"}))
            generated.append((fake_tensor, 1, {**parent, "view": "generated"}))
            sources.append(
                {
                    "metadata": parent,
                    "label": 0,
                    "generation": {
                        "source_label": 0,
                        "generated_label": 1,
                        "fallback": False,
                        "landmarks": parent["attributes"]["landmarks"],
                        "mask_sha256": pixel_digest(mask),
                        "original_sha256": pixel_digest(real_tensor.numpy()),
                        "generated_sha256": pixel_digest(fake_tensor.numpy()),
                        "rng_before_sha256": hashlib.sha256(
                            json.dumps(source_rng).encode()
                        ).hexdigest(),
                        "rng_after_sha256": rng_digest(),
                    },
                }
            )
        pixels, labels, metadata = zip(*(originals + generated), strict=True)
        return {
            "pixel_values": torch.stack(pixels),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": list(metadata),
            "sources": sources,
            "generation_rng": {"before": before, "after": rng_state()},
        }
