"""Native self-blending component used only by the ProDet training recipe."""

import random

import cv2
import numpy as np

from detectors._shared.augmentation import AugmentationLibrary
from .masks import FaceHullMask


class SoftBlendMask:
    """Sample the native two-stage blur without altering RNG ordering."""

    @staticmethod
    def create(mask):
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


class SelfBlendingGenerator:
    def __init__(self):
        from .transforms import RandomDownScale

        A = AugmentationLibrary.load()
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
        A = AugmentationLibrary.load()
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
        mask = FaceHullMask.create(landmarks, image, kind)
        source = image.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))["image"]
        else:
            image = self.source_transforms(image=image.astype(np.uint8))["image"]
        source, mask = self.randaffine(source, mask)
        mask = SoftBlendMask.create(mask)
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

