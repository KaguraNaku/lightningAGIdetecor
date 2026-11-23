"""Author image operations, independent of model construction and data storage."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from detectors._shared.augmentation import load_albumentations

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
PROFILE = "effort-author-96f5dea-albumentations-1.3.1"


def author_augmentation():
    A = load_albumentations()
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.1),
            A.GaussNoise(p=0.1),
            A.MotionBlur(p=0.1),
            A.CLAHE(p=0.1),
            A.ChannelShuffle(p=0.1),
            A.Cutout(p=0.1),
            A.RandomGamma(p=0.3),
            A.GlassBlur(p=0.3),
        ]
    )


@dataclass
class EffortTransform:
    training: bool = False
    size: int = 224
    mean: tuple = MEAN
    std: tuple = STD

    def __post_init__(self):
        if type(self.size) is not int or self.size < 1:
            raise ValueError("Effort image size must be a positive integer")
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or any(s <= 0 for s in self.std)
        ):
            raise ValueError(
                "Effort normalization requires three finite means/positive stds"
            )
        self.augmentation = author_augmentation() if self.training else None

    def __call__(self, image):
        pixels = cv2.resize(
            np.array(image.convert("RGB")),
            (self.size, self.size),
            interpolation=cv2.INTER_CUBIC,
        )
        if self.augmentation is not None:
            pixels = self.augmentation(image=pixels)["image"]
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        tensor = tensor.to(dtype=torch.float32).div(255)
        mean = tensor.new_tensor(self.mean)[:, None, None]
        std = tensor.new_tensor(self.std)[:, None, None]
        return tensor.sub(mean).div(std)


def preprocessing_config(size):
    return {"profile": PROFILE, "size": size, "mean": list(MEAN), "std": list(STD)}


def make_transform(config, *, training=False):
    if config.get("profile") != PROFILE:
        raise ValueError("Unsupported Effort preprocessing profile")
    return EffortTransform(
        training, config["size"], tuple(config["mean"]), tuple(config["std"])
    )
