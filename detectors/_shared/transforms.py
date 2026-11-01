"""Serializable RGB preprocessing from the local DeepfakeBench image pipeline."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

PROFILE = "dfb-local-rgb-cubic-albu131-v1"
DEFAULT_AUGMENTATION = {
    "flip_prob": 0.5,
    "rotate_prob": 0.5,
    "rotate_limit": [-10, 10],
    "blur_prob": 0.5,
    "blur_limit": [3, 7],
    "brightness_limit": [-0.1, 0.1],
    "contrast_limit": [-0.1, 0.1],
    "quality_lower": 40,
    "quality_upper": 100,
}


def preprocessing_config(size, mean, std, *, augmentation=True, data_aug=None):
    return {
        "profile": PROFILE,
        "size": size,
        "mean": list(mean),
        "std": list(std),
        "augmentation": augmentation,
        "data_aug": {**DEFAULT_AUGMENTATION, **(data_aug or {})},
    }


def build_augmentation(config):
    import albumentations as A

    if A.__version__ != "1.3.1":
        raise RuntimeError("This preprocessing profile requires albumentations==1.3.1")
    return A.Compose(
        [
            A.HorizontalFlip(p=config["flip_prob"]),
            A.Rotate(limit=config["rotate_limit"], p=config["rotate_prob"]),
            A.GaussianBlur(blur_limit=config["blur_limit"], p=config["blur_prob"]),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=config["brightness_limit"],
                        contrast_limit=config["contrast_limit"],
                    ),
                    A.FancyPCA(),
                    A.HueSaturationValue(),
                ],
                p=0.5,
            ),
            A.ImageCompression(
                quality_lower=config["quality_lower"],
                quality_upper=config["quality_upper"],
                p=0.5,
            ),
        ]
    )


@dataclass
class ImageTransform:
    config: dict
    training: bool = False

    def __post_init__(self):
        if self.config.get("profile") != PROFILE:
            raise ValueError("Unsupported image preprocessing profile")
        size = self.config.get("size")
        if type(size) is not int or size < 1:
            raise ValueError("Image size must be a positive integer")
        mean, std = self.config["mean"], self.config["std"]
        if (
            len(mean) != 3
            or len(std) != 3
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or min(std) <= 0
        ):
            raise ValueError("Normalization needs three finite means and positive stds")
        self.augmentation = (
            build_augmentation(self.config["data_aug"])
            if self.training and self.config["augmentation"]
            else None
        )

    def __call__(self, image):
        pixels = cv2.resize(
            np.asarray(image.convert("RGB")),
            (self.config["size"],) * 2,
            interpolation=cv2.INTER_CUBIC,
        )
        if self.augmentation is not None:
            pixels = self.augmentation(image=pixels)["image"]
        tensor = (
            torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            .float()
            .div(255)
        )
        return tensor.sub(tensor.new_tensor(self.config["mean"])[:, None, None]).div(
            tensor.new_tensor(self.config["std"])[:, None, None]
        )
