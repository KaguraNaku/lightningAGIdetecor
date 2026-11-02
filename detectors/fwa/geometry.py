"""Native FWA similarity/alignment and pixelwise warp-back composition."""

from importlib.resources import files
import json
import os
from pathlib import Path

import cv2
import numpy as np

from data.manifest import sha256


def canonical_landmarks():
    values = json.loads(
        files("configs").joinpath("migrations/fwa_landmarks.json").read_text()
    )
    return np.stack([values["mean_face_x"], values["mean_face_y"]], axis=1)


def similarity(source, destination, estimate_scale=True):
    count, dimensions = source.shape
    source_mean, destination_mean = source.mean(0), destination.mean(0)
    centered_source, centered_destination = (
        source - source_mean,
        destination - destination_mean,
    )
    covariance = np.dot(centered_destination.T, centered_source) / count
    signs = np.ones((dimensions,), dtype=np.double)
    if np.linalg.det(covariance) < 0:
        signs[dimensions - 1] = -1
    result = np.eye(dimensions + 1, dtype=np.double)
    left, singular, right = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        return np.nan * result
    if rank == dimensions - 1:
        if np.linalg.det(left) * np.linalg.det(right) > 0:
            result[:dimensions, :dimensions] = np.dot(left, right)
        else:
            previous = signs[dimensions - 1]
            signs[dimensions - 1] = -1
            result[:dimensions, :dimensions] = np.dot(
                left, np.dot(np.diag(signs), right)
            )
            signs[dimensions - 1] = previous
    else:
        # Preserve the source's V.T placement, including its reflection behavior.
        result[:dimensions, :dimensions] = np.dot(left, np.dot(np.diag(signs), right.T))
    scale = (
        1.0 / centered_source.var(0).sum() * np.dot(singular, signs)
        if estimate_scale
        else 1.0
    )
    result[:dimensions, dimensions] = destination_mean - scale * np.dot(
        result[:dimensions, :dimensions], source_mean.T
    )
    result[:dimensions, :dimensions] *= scale
    return result


def aligned_face(image, matrix, size=256):
    return cv2.warpAffine(image, matrix * size, (size, size))


def aligned_points(matrix, points, size=256):
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1).T
    return np.dot(matrix * size, homogeneous)[:2].T


def warp_back(image, face, matrix, size=256):
    new_face = np.clip(face, 0, 255).astype(image.dtype)
    warped = np.zeros_like(image)
    native_matrix = matrix * size + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cv2.warpAffine(
        new_face,
        native_matrix,
        (image.shape[1], image.shape[0]),
        warped,
        cv2.WARP_INVERSE_MAP,
        cv2.BORDER_TRANSPARENT,
    )
    # Native composition uses a separate positive-value mask per color channel.
    mask = (warped > 0).astype(np.uint8)
    return image * (1 - mask) + warped * mask, mask


class LazyDlibAlignment:
    """One predictor owner per worker; loading never happens during imports."""

    def __init__(self, receipt):
        self.receipt = receipt
        self._pid = None
        self._detector = self._predictor = None

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_detector": None, "_predictor": None}

    def __call__(self, image):
        if self._pid != os.getpid():
            import dlib

            if dlib.__version__ != "20.0.0":
                raise RuntimeError("Native FWA generation requires dlib==20.0.0")
            path = Path(self.receipt["path"])
            if sha256(path) != self.receipt["sha256"]:
                raise ValueError(
                    "FWA landmark predictor differs from its saved identity"
                )
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(str(path))
            self._pid = os.getpid()
        image = np.uint8(image)
        result = []
        for face in self._detector(image, 0):
            try:
                shape = self._predictor(image, face)
            except Exception:
                shape = self._predictor(image, face.rect)
            points = np.zeros((68, 2), dtype="int")
            for index in range(68):
                points[index] = shape.part(index).x, shape.part(index).y
            matrix = similarity(points[17:], canonical_landmarks(), True)[:2]
            result.append((matrix, points))
        return result
