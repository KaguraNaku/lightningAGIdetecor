"""FFBlend face generation with native raw boundaries and discarded RNG draws."""

import random

import cv2
import numpy as np
import torch

from detectors.fwa.generation import FWAGenerator, pixel_digest
from detectors.fwa.masks import face_mask, deform_mask
from .geometry import warp_face


def color_transfer(source, destination, mask):
    rows, columns = np.where(mask != 0)
    selected_source = source[rows, columns].astype(np.float32)
    selected_destination = destination[rows, columns].astype(np.float32)
    mean_source = np.mean(selected_source, axis=0)
    std_source = np.std(selected_source, axis=0)
    mean_destination = np.mean(selected_destination, axis=0)
    std_destination = np.std(selected_destination, axis=0)
    transferred = (selected_destination - mean_destination) * (
        std_source / std_destination
    ) + mean_source
    transferred = np.clip(transferred, 0, 255)
    result = np.copy(destination)
    result[rows, columns] = transferred.astype(np.uint8)
    return result


def feather_blend(foreground, background, mask, amount=0.2):
    values = np.repeat((mask / 255.0)[:, :, None], 3, axis=2)
    indices = np.where(values != 0)
    points = np.hstack((indices[1][:, None], indices[0][:, None]))
    if points.size == 0:
        # Source returns the entire corrected foreground for an empty mask.
        return foreground
    size = np.max(points, axis=0) - np.min(points, axis=0)
    amount = amount * np.max(size)
    hull = cv2.convexHull(points)
    distances = np.zeros(points.shape[0])
    for index in range(points.shape[0]):
        distances[index] = cv2.pointPolygonTest(
            hull, (int(points[index, 0]), int(points[index, 1])), True
        )
    weights = np.clip(distances / amount, 0, 1)
    foreground, background = foreground.astype(float), background.astype(float)
    result = np.copy(background)
    result[indices[0], indices[1]] = (
        weights[:, None] * foreground[indices[0], indices[1]]
        + (1 - weights[:, None]) * background[indices[0], indices[1]]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def boundary(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    alpha = cv2.GaussianBlur(mask, (3, 3), 0) / 255.0
    return 4 * alpha * (1.0 - alpha)


def assemble_pairs(pairs):
    """Whole fake-all/real-all pairs, jointly shuffled with masks and parents.

    Each pair is (fake_pixels, real_pixels, boundary, source_label). The returned
    source-index/view order lets a method collator attach exact parent metadata.
    """
    views = [
        (fake, target, 1, index, "generated")
        for index, (fake, _, target, _) in enumerate(pairs)
    ] + [
        (real, torch.zeros_like(target), label, index, "original")
        for index, (_, real, target, label) in enumerate(pairs)
    ]
    random.shuffle(views)
    return {
        "pixel_values": torch.stack([view[0] for view in views]),
        "masks": torch.stack([view[1] for view in views]),
        "labels": torch.tensor([view[2] for view in views]),
        "view_order": [(view[3], view[4]) for view in views],
    }


class FaceXrayGenerator:
    @staticmethod
    def postprocess(image):
        swapped = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        FWAGenerator.blended_aug(
            swapped
        )  # Native output is discarded; RNG is retained.
        tensor = (
            torch.from_numpy(np.ascontiguousarray(np.uint8(swapped).transpose(2, 0, 1)))
            .float()
            .div(255)
        )
        return tensor.sub_(0.5).div_(0.5)

    def generate(self, foreground, foreground_points, background, background_points):
        foreground = np.array(FWAGenerator.original_aug(foreground))
        foreground_points = np.array(foreground_points, dtype=np.int32)
        if callable(background):
            # Dataset IO may be lazy so a failed background read occurs AFTER
            # native foreground augmentation and its RNG consumption.
            background = background()
        background = np.array(FWAGenerator.original_aug(background))
        background_points = np.array(background_points, dtype=np.int32)
        foreground_mask = face_mask(foreground_points, foreground, deform=False)
        background_mask = face_mask(background_points, background, deform=True)
        background_mask = deform_mask(background_mask, 20)
        warped, foreground_mask, receipt = warp_face(
            foreground, foreground_points, background_points, foreground_mask
        )
        try:
            background_mask[foreground_mask == 0] = 0
            corrected = color_transfer(background, warped, background_mask)
            generated = feather_blend(corrected, background, background_mask)
        except Exception:
            # Native retries the same color/blend calls without regenerating data.
            corrected = color_transfer(background, warped, background_mask)
            generated = feather_blend(corrected, background, background_mask)
        target = boundary(background_mask)
        fake, real = self.postprocess(generated), self.postprocess(background)
        target = torch.from_numpy(target).unsqueeze(2).permute(2, 0, 1)
        return (
            fake,
            real,
            target,
            {
                **receipt,
                "mask_sha256": pixel_digest(background_mask),
                "boundary_sha256": pixel_digest(target.numpy()),
                "generated_sha256": pixel_digest(fake.numpy()),
                "original_sha256": pixel_digest(real.numpy()),
                "empty_mask_returns_foreground": not bool(np.any(background_mask)),
            },
        )
