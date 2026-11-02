"""Original LSDA per-frame augmentation, followed by native tensor normalization."""

import albumentations as A
import cv2
import numpy as np
import torch


class RandomDownScale(A.ImageOnlyTransform):
    def apply(self, image, **params):
        height, width, _ = image.shape
        ratios = [2, 4]
        ratio = ratios[np.random.randint(len(ratios))]
        reduced = cv2.resize(
            image,
            (int(width / ratio), int(height / ratio)),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)


def build_augmentation():
    if A.__version__ != "1.3.1":
        raise RuntimeError("Native LSDA augmentation requires albumentations==1.3.1")
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            RandomDownScale(p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
        ],
        additional_targets={f"image{index}": "image" for index in range(1, 5)},
        p=1.0,
    )


class LSDATransform:
    """Receive an already RGB/cubic-resized256 PIL image, as the source does."""

    def __init__(self):
        self.augmentation = build_augmentation()

    def __call__(self, image):
        pixels = self.augmentation(image=np.asarray(image))["image"]
        pixels = np.asarray(pixels, dtype=np.uint8)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            .to(dtype=torch.get_default_dtype())
            .div(255)
        )
        return tensor.sub_(tensor.new_tensor([0.5] * 3)[:, None, None]).div_(
            tensor.new_tensor([0.5] * 3)[:, None, None]
        )
