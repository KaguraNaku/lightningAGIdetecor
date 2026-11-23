"""Selected native whole-face SBI plus soft BI synthesis, before collation.

The caller owns source pairing, hash-bound landmarks and nearest-face selection.
This module never chooses replacement sources when synthesis fails.
"""

import random

import numpy as np
import torch

from detectors._shared.augmentation import load_albumentations
from .blending import SelfBlendingGenerator, SoftBlendMask
from .masks import FaceHullMask


DOMAINS = (("real", 0), ("blendfake", 1), ("bifake", 3), ("deepfake", 2))


def select_foreground(background, neighbors):
    """Native top-three draw from an already prepared ordered neighbor table.

    Sidecar loading/filtering, hashes and train-split validation belong to the
    dataset adapter. In particular this does not reproduce init_nearest's
    mutation-while-iterating filter or silently repair that source behavior.
    """
    if background in neighbors:
        return random.choice(neighbors[background][:3])
    return background


def normalized(image):
    """Native ToTensor followed by mean/std=.5, distinct from SBI's ImageNet."""
    return (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        .float()
        .div(255)
        .sub(0.5)
        .div(0.5)
    )


class ProDetGenerator:
    """The selected prodet.yaml recipe: whole_face, BI on, soft_bi=True."""

    def __init__(self):
        self.sbi = SelfBlendingGenerator()
        A = load_albumentations()
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
            additional_targets={
                "image1": "image",
                "image2": "image",
                "image3": "image",
            },
            p=1.0,
        )

    @staticmethod
    def color_transfer(source, destination, mask):
        # Source uses float32 masked statistics and no epsilon/clamping of std.
        indices = np.where(mask != 0)
        src = source[indices[0], indices[1]].astype(np.float32)
        dst = destination[indices[0], indices[1]].astype(np.float32)
        mean_src, std_src = np.mean(src, axis=0), np.std(src, axis=0)
        mean_dst, std_dst = np.mean(dst, axis=0), np.std(dst, axis=0)
        shifted = (dst - mean_dst) * (std_src / std_dst) + mean_src
        shifted = np.clip(shifted, 0, 255)
        # Fail rather than silently casting native NaNs into replacement pixels.
        if not np.isfinite(shifted).all() or not len(indices[0]):
            raise ValueError(
                "ProDet BI color transfer produced nonfinite/empty masked pixels"
            )
        result = np.copy(destination)
        result[indices[0], indices[1]] = shifted.astype(np.uint8)
        return result

    def bi_blending(self, background, foreground, landmarks):
        if np.random.rand() < 0.25:
            landmarks = landmarks[:68]
        # random_get_hull(None) selects all four hulls for whole_face.
        mask = FaceHullMask.create(landmarks, background, random.choice([0, 1, 2, 3]))
        source, mask = self.sbi.randaffine(foreground.copy(), mask)
        source = self.color_transfer(background.copy(), source, mask)
        blurred = SoftBlendMask.create(mask)
        ratio = [0.25, 0.5, 0.75, 1, 1, 1][np.random.randint(6)]
        blurred *= ratio
        generated = blurred * source + (1 - blurred) * background
        if not np.isfinite(generated).all() or not np.isfinite(blurred).all():
            raise ValueError("ProDet BI blending produced nonfinite pixels/mask")
        return (
            background.astype(np.uint8),
            generated.astype(np.uint8),
            blurred.squeeze(2),
        )

    def generate(self, real, deepfake, landmarks, *, load_foreground):
        """Choose/load the BI foreground AFTER SBI, preserving native RNG order.

        ``load_foreground`` performs the caller's nearest-face random choice and
        decoding. Choosing the foreground before this call changes synthesis RNG.
        """
        for image in (real, deepfake):
            if (
                not isinstance(image, np.ndarray)
                or image.shape != (256, 256, 3)
                or image.dtype != np.uint8
            ):
                raise ValueError(
                    "ProDet synthesis requires decoded uint8 RGB256 inputs"
                )
        if (
            not isinstance(landmarks, np.ndarray)
            or landmarks.shape not in ((68, 2), (81, 2))
            or not np.isfinite(landmarks).all()
        ):
            raise ValueError(
                "ProDet requires finite native 68/81-point 256-coordinate landmarks"
            )
        if not callable(load_foreground):
            raise ValueError(
                "ProDet foreground selection must run after SBI through a callback"
            )
        landmarks = landmarks.astype(np.int32)
        # Native SBI_API(..., aug=False, return_mask=True): no pair augmentation.
        original, blend, sbi_mask = self.sbi.self_blending(
            real.copy(), landmarks.copy()
        )
        foreground = load_foreground()
        if (
            not isinstance(foreground, np.ndarray)
            or foreground.shape != (256, 256, 3)
            or foreground.dtype != np.uint8
        ):
            raise ValueError(
                "ProDet foreground loader must return decoded uint8 RGB256"
            )
        original, bi, bi_mask = self.bi_blending(
            original.copy(), foreground.copy(), landmarks.copy()
        )
        transformed = self.transforms(
            image=blend.astype("uint8"),
            image1=original.astype("uint8"),
            image2=deepfake.astype("uint8"),
            image3=bi.astype("uint8"),
        )
        images = dict(
            zip(
                (name for name, _ in DOMAINS),
                (
                    transformed["image1"],
                    transformed["image"],
                    transformed["image3"],
                    transformed["image2"],
                ),
                strict=True,
            )
        )
        if not np.isfinite(sbi_mask).all():
            raise ValueError("ProDet SBI produced a nonfinite mask")
        return {
            **{name: (normalized(images[name]), label) for name, label in DOMAINS},
            "masks": {"sbi": sbi_mask, "bi": bi_mask},
        }


class ProDetQuadCollator:
    """Spawn-safe replacement for the native process-global config collator.

    Only the selected equal/full sequence recipe is implemented. Dataset wiring
    must supply aligned metadata for each generated view before using this.
    """

    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError(
                "ProDet quadruples must be nonempty with no discarded sources"
            )
        pixels, domain_labels, metadata = [], [], []
        for name, label in DOMAINS:
            images = []
            for sample in samples:
                image, actual_label = sample[name]
                if (
                    not isinstance(image, torch.Tensor)
                    or image.device.type != "cpu"
                    or image.dtype != torch.float32
                    or image.shape != (3, 256, 256)
                    or not torch.isfinite(image).all()
                    or type(actual_label) is not int
                    or actual_label != label
                ):
                    raise ValueError(
                        f"Invalid ProDet {name} view tensor or native domain label"
                    )
                record = sample["metadata"][name]
                if not isinstance(record, dict) or not record.get("id"):
                    raise ValueError(
                        "ProDet views require explicit source identity metadata"
                    )
                images.append(image)
                metadata.append(dict(record))
            pixels.append(torch.stack(images))
            domain_labels.append(
                torch.tensor([label] * len(samples), dtype=torch.float32)
            )
        labels = torch.cat(domain_labels)
        return {
            "pixel_values": torch.cat(pixels),
            "prodet_labels": labels,
            "labels": labels.ne(0).long(),
            "metadata": metadata,
        }
