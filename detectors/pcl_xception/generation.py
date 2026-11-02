"""Active I2G generation only: clipped/reordered BG landmarks and raw blend mask."""

import random

import numpy as np
import torch

from detectors.fwa.generation import augmentation_library, pixel_digest
from detectors.sbi.generation import blend_mask, SBIGenerator
from detectors.sbi.masks import hull_mask


FOREHEAD_ORDER = [77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]


def normalized(image):
    value = (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        .float()
        .div(255)
    )
    return value.sub_(0.5).div_(0.5)


def color_transfer(source, destination, mask):
    rows, columns = np.where(mask != 0)
    selected_source = source[rows, columns].astype(np.float32)
    selected_destination = destination[rows, columns].astype(np.float32)
    mean_source, std_source = (
        np.mean(selected_source, axis=0),
        np.std(selected_source, axis=0),
    )
    mean_destination, std_destination = (
        np.mean(selected_destination, axis=0),
        np.std(selected_destination, axis=0),
    )
    transferred = (selected_destination - mean_destination) * (
        std_source / std_destination
    ) + mean_source
    result = np.copy(destination)
    result[rows, columns] = np.clip(transferred, 0, 255).astype(np.uint8)
    return result


class I2GGenerator:
    def __init__(self):
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

    def generate(self, background, foreground, background_landmarks):
        points = np.clip(background_landmarks, 0, 255).copy()
        if points.shape != (81, 2):
            raise ValueError(
                "Configured I2G requires81 background points before forehead reordering"
            )
        points[68:] = points[FOREHEAD_ORDER].copy()
        reordered = points.copy()
        if np.random.rand() < 0.25:
            points = points[:68]
        mask = hull_mask(points, background, random.choice([0, 1, 2, 3]))
        source, mask = SBIGenerator.randaffine(foreground.copy(), mask)
        source = color_transfer(background.copy(), source, mask)
        mask = blend_mask(mask)
        # Source still draws randint3 although every blend ratio is1.
        mask *= [1, 1, 1][np.random.randint(3)]
        generated = (mask * source + (1 - mask) * background.copy()).astype(np.uint8)
        transformed = self.transforms(
            image=generated, image1=background.astype(np.uint8)
        )
        fake, real = normalized(transformed["image"]), normalized(transformed["image1"])
        target = torch.from_numpy(np.ascontiguousarray(mask.squeeze(2))).unsqueeze(0)
        return (
            fake,
            real,
            target,
            {
                "background_landmarks_sha256": pixel_digest(reordered),
                "mask_sha256": pixel_digest(target.numpy()),
                "generated_sha256": pixel_digest(fake.numpy()),
                "original_sha256": pixel_digest(real.numpy()),
                "background_landmarks_clipped": True,
                "background_forehead_reordered": True,
                "source_photometric_transform_active": False,
                "hflip_active": False,
            },
        )


def assemble_pairs(pairs):
    """Return real-all then fake-all without shuffle, preserving raw source masks."""
    fake, real, mask = zip(*pairs, strict=True)
    return {
        "pixel_values": torch.cat([torch.stack(real), torch.stack(fake)], dim=0),
        "labels": torch.tensor([0] * len(real) + [1] * len(fake)),
        "masks": torch.cat(
            [torch.zeros_like(torch.stack(mask)), torch.stack(mask)], dim=0
        ),
    }
