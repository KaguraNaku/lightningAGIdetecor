"""Native selected pairDataset augmentation with shared landmarks and masks."""

import numpy as np
import torch


AUGMENTATION = {
    "flip_prob": 0.5,
    "rotate_prob": 0,
    "rotate_limit": [-10, 10],
    "blur_prob": 0.5,
    "blur_limit": [3, 7],
    "brightness_limit": [-0.1, 0.1],
    "contrast_limit": [-0.1, 0.1],
    "quality_lower": 40,
    "quality_upper": 100,
}


class SLADDPairTransform:
    def __init__(self):
        import albumentations as A

        if A.__version__ != "1.3.1":
            raise RuntimeError(
                "SLADD paired augmentation requires albumentations==1.3.1"
            )
        self.library = A
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=AUGMENTATION["flip_prob"]),
                A.Rotate(
                    limit=AUGMENTATION["rotate_limit"], p=AUGMENTATION["rotate_prob"]
                ),
                A.GaussianBlur(
                    blur_limit=AUGMENTATION["blur_limit"], p=AUGMENTATION["blur_prob"]
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=AUGMENTATION["brightness_limit"],
                            contrast_limit=AUGMENTATION["contrast_limit"],
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=0.5,
                ),
                A.ImageCompression(
                    quality_lower=AUGMENTATION["quality_lower"],
                    quality_upper=AUGMENTATION["quality_upper"],
                    p=0.5,
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

    def __call__(self, image, landmarks, mask):
        arguments = {
            "image": image,
            "keypoints": landmarks,
            "keypoint_params": self.library.KeypointParams(format="xy"),
        }
        mask = mask.squeeze(2)
        if mask.max() > 0:
            arguments["mask"] = mask
        transformed = self.transform(**arguments)
        points = np.array(transformed["keypoints"])
        mask = transformed.get("mask", mask)
        pixels = (
            torch.from_numpy(
                np.ascontiguousarray(transformed["image"].transpose(2, 0, 1))
            )
            .float()
            .div(255)
        )
        pixels = pixels.sub(pixels.new_tensor([0.5] * 3)[:, None, None]).div(
            pixels.new_tensor([0.5] * 3)[:, None, None]
        )
        return pixels, torch.from_numpy(points), torch.from_numpy(mask)
