"""Native per-frame pixels with a shared clip seed and explicit RNG repair."""

import random

import albumentations as A
import numpy as np
import torch


class I3DClipTransform:
    """Restore worker Python/NumPy state instead of native entropy reseeding.

    Conditional per-clip pixel agreement must be audited separately. This policy
    intentionally does not reproduce the native process RNG after augmentation.
    """

    def __init__(self, *, training):
        if A.__version__ != "1.3.1":
            raise RuntimeError("Native I3D augmentation requires albumentations1.3.1")
        self.training = training
        self.augmentation = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=(-10, 10), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=0.5,
                ),
                A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            ],
            keypoint_params=None,
        )

    def __call__(self, pixels, *, augmentation_seed):
        if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 3):
            raise ValueError("I3D preprocessing requires resized RGB224 uint8 pixels")
        if self.training:
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                random.seed(augmentation_seed)
                np.random.seed(augmentation_seed)
                pixels = self.augmentation(image=pixels)["image"]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        # Match torchvision's contiguous CHW uint8 -> default float -> /255,
        # followed by Normalize's in-place subtraction and division.
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).contiguous()
        tensor = tensor.to(dtype=torch.get_default_dtype()).div(255)
        tensor = tensor.clone()
        return tensor.sub_(0.5).div_(0.5)
