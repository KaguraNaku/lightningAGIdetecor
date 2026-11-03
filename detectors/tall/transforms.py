"""Two explicit local TALL data recipes; ImageNet normalization in both.

The formal train.py entry selects its generic dataset, while the separately
defined TALLDataset adds cutout in every stage. These must never be silently
merged. Both use the same native augmentation operators when their role enables
augmentation. Worker RNG restoration is an explicit remake runtime policy.
"""

import random

import numpy as np
import torch

from detectors._shared.transforms import DEFAULT_AUGMENTATION, build_augmentation

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PROFILES = {
    "native_entrypoint": {
        "augmentation_stages": ("train", "validation"),
        "cutout": False,
        "reference": "Actual train.py generic dataset fallback for dataset_type=tall",
    },
    "specialized_cutout": {
        "augmentation_stages": ("train",),
        "cutout": True,
        "reference": "Explicit TALLDataset body;requires deliberate selector/collator wiring repair",
    },
}


class TALLClipTransform:
    def __init__(self, *, augment):
        if type(augment) is not bool:
            raise ValueError("TALL augmentation flag must be boolean")
        self.augmentation = (
            build_augmentation(DEFAULT_AUGMENTATION) if augment else None
        )

    def __call__(self, pixels, *, augmentation_seed):
        if (
            pixels.dtype != np.uint8
            or pixels.shape != (224, 224, 3)
            or torch.get_default_dtype() != torch.float32
        ):
            raise ValueError(
                "TALL requires resized RGB224 uint8 and FP32 tensor conversion"
            )
        if self.augmentation is not None:
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                random.seed(augmentation_seed)
                np.random.seed(augmentation_seed)
                pixels = self.augmentation(image=pixels)["image"]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        # Literal torchvision ToTensor followed by Normalize's in-place recipe.
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).contiguous()
        tensor = tensor.to(dtype=torch.get_default_dtype()).div(255).clone()
        return tensor.sub_(tensor.new_tensor(MEAN)[:, None, None]).div_(
            tensor.new_tensor(STD)[:, None, None]
        )


def apply_cutout(clips, *, rng):
    """Native normalized-space -1 rectangle shared by all four RGB frames."""
    if clips.shape != (4, 3, 224, 224) or clips.dtype != torch.float32:
        raise ValueError("TALL cutout requires a complete FP32 T4/RGB224 clip")
    height, width = clips.shape[-2:]
    x, y = int(rng.randint(width)), int(rng.randint(height))
    x1, x2 = int(np.clip(x - 8, 0, width)), int(np.clip(x + 8, 0, width))
    y1, y2 = int(np.clip(y - 8, 0, height)), int(np.clip(y + 8, 0, height))
    clips[:, :, y1:y2, x1:x2] = -1
    return {"xy": [x, y], "bounds": [x1, y1, x2, y2], "grid_size": 16}
