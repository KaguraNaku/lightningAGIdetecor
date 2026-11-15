"""Importable native downscale transform for spawned ProDet loader workers."""

import albumentations as A
import cv2
import numpy as np


class RandomDownScale(A.core.transforms_interface.ImageOnlyTransform):
    def apply(self, image, **params):
        height, width, _ = image.shape
        ratio = [2, 4][np.random.randint(2)]
        down = cv2.resize(
            image,
            (int(width / ratio), int(height / ratio)),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)
