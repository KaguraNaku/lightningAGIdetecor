"""Author PIL augmentation; torchvision is imported only for selected GenD training."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class GenDTransform:
    config: dict
    training: bool = False

    def __post_init__(self):
        from torchvision.transforms import v2 as T
        from torchvision.transforms import InterpolationMode

        if self.config["profile"] != "gend-author-pil-v1":
            raise ValueError("Unknown GenD author preprocessing profile")
        # In the author Dataset, augmentation acts on PIL before native CLIP preprocessing.
        self.augmentation = (
            T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    T.RandomApply(
                        [T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.1
                    ),
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                    T.JPEG((40, 100)),
                ]
            )
            if self.training
            else None
        )
        self.preprocessing = T.Compose(
            [
                T.Resize(self.config["size"], interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(self.config["size"]),
            ]
        )

    def __call__(self, image):
        image = image.convert("RGB")
        if self.augmentation is not None:
            image = self.augmentation(image)
        pixels = np.array(self.preprocessing(image), copy=True)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255)
        return tensor.sub(tensor.new_tensor(self.config["mean"])[:, None, None]).div(
            tensor.new_tensor(self.config["std"])[:, None, None]
        )
