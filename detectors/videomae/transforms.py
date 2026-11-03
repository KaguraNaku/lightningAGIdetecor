"""Native landmark-conditioned clip pixels with explicit worker RNG repair."""

import random

import albumentations as A
import numpy as np
import torch


class VideoMAEClipTransform:
    def __init__(self, *, augment):
        if A.__version__ != "1.3.1":
            raise RuntimeError("Native VideoMAE pixels require albumentations1.3.1")
        self.augment = augment
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-10, 10), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
                A.FancyPCA(),
                A.HueSaturationValue(),
            ], p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
        ], keypoint_params=A.KeypointParams(format="xy"))

    def __call__(self, pixels, landmarks, *, augmentation_seed):
        if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 3):
            raise ValueError("VideoMAE preprocessing requires resized RGB224 uint8 pixels")
        if self.augment:
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                random.seed(augmentation_seed)
                np.random.seed(augmentation_seed)
                pixels = self.augmentation(
                    image=pixels, keypoints=landmarks,
                    keypoint_params=A.KeypointParams(format="xy"),
                )["image"]
            finally:
                # Native entropy-reseeds after every frame. This deliberate
                # repair preserves worker RNG; conditional pixels stay native.
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).contiguous()
        tensor = tensor.to(dtype=torch.get_default_dtype()).div(255).clone()
        # Native __getitem__ overwrites augmented landmarks with the ORIGINAL
        # NPY tensor. Keep this quirk even though the Detector ignores the field.
        return tensor.sub_(0.5).div_(0.5), torch.from_numpy(landmarks)
