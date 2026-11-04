"""Native SLADD region/deformation/three blending paths, without the networks.

Keep source row-noise indexing, bool mask union, feather classification rather
than distance, and quantization. Image tensors are already normalized RGB;
native color helpers still use the original OpenCV BGR conversion constants.
"""

import math

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from .colors import color_transfer, faceswap_color


def distance(first, second):
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def landmark_keys(points):
    values = [
        (points[36] + points[39]) * 0.5,
        (points[42] + points[45]) * 0.5,
        points[33],
        points[48],
        points[54],
        points[36],
        points[39],
        points[42],
        points[45],
    ]
    return [tuple(value.astype("int32")) for value in values]


def eye_region(image, keys, choice):
    indices = {"l": (5, 7), "r": (7, 9), "b": (0, 2)}[choice]
    first, second = keys[slice(*indices)]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, first, second, color=(1), thickness=2)
    dilation = int(distance(first, second) // 4)
    if choice != "b":
        dilation *= 4
    return binary_dilation(line, iterations=dilation)


def nose_region(image, keys):
    first, second = keys[:2]
    end = int((first[0] + second[0]) / 2), int((first[1] + second[1]) / 2)
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, keys[2], end, color=(1), thickness=2)
    return binary_dilation(line, iterations=int(distance(first, second) // 4))


def mouth_region(image, keys):
    first, second = keys[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, first, second, color=(1), thickness=2)
    return binary_dilation(line, iterations=int(distance(first, second) // 3))


def random_mask(mask, resolution=256):
    from skimage import measure

    left, right, up, down = [np.random.randint(10, 60) for _ in range(4)]
    result = np.zeros(mask.shape)
    mask = np.where(mask > 0.1, 1, 0)
    properties = measure.regionprops(mask)
    if not properties:
        return result
    row, column = [int(round(value)) for value in properties[0].centroid]
    result[
        max(row - left, 0) : min(row + right, resolution - 1),
        max(column - up, 0) : min(row + down, resolution - 1),
    ] = 1
    result *= mask
    return result


def random_deform(mask, rows_count=5, columns_count=5, mean=0, std=10):
    from skimage.transform import PiecewiseAffineTransform, warp

    height, width = mask.shape[:2]
    rows = np.linspace(0, height - 1, rows_count).astype(np.int32)
    columns = np.linspace(0, width - 1, columns_count).astype(np.int32)
    rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
    # Native second draw also mutates rows. Do not replace it with columns +=.
    rows += np.random.normal(mean, std, size=columns.shape).astype(np.int32)
    rows, columns = np.meshgrid(rows, columns)
    anchors = np.vstack([rows.flat, columns.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == columns_count * rows_count
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:, 0], 0, height - 1, deformed[:, 0])
    np.clip(deformed[:, 1], 0, width - 1, deformed[:, 1])
    transform = PiecewiseAffineTransform()
    transform.estimate(anchors, deformed.astype(np.int32))
    warped = warp(mask, transform)
    warped *= mask
    return cv2.GaussianBlur(warped.astype(float), (5, 5), 3)


def parse_region(image, region, landmarks, fake_mask):
    keys = landmark_keys(landmarks)
    if region in (0, 1, 2):
        mask = eye_region(image, keys, ("l", "r", "b")[region])
    elif region == 3:
        mask = nose_region(image, keys)
    elif region == 4:
        mask = mouth_region(image, keys)
    elif region in (5, 6, 7):
        mask = nose_region(image, keys) + eye_region(
            image, keys, ("l", "r", "b")[region - 5]
        )
    elif region == 8:
        mask = nose_region(image, keys) + mouth_region(image, keys)
    elif region == 9:
        mask = (
            eye_region(image, keys, "b")
            + nose_region(image, keys)
            + mouth_region(image, keys)
        )
    else:
        mask = random_mask(fake_mask)
    return random_deform(mask, 5, 5) * 1.0


def mask_postprocess(mask):
    draw = np.random.rand()
    if draw < 0.3:
        size = 2 * np.random.randint(1, 10) + 1
        mask = cv2.erode(mask, np.ones((size, size), np.uint8))
    elif draw < 0.6:
        size = 2 * np.random.randint(1, 10) + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    if np.random.rand() < 0.9:
        size = 2 * np.random.randint(1, 10) - 1
        mask = cv2.GaussianBlur(mask, (size, size), 0)
    return mask


def feather_blend(source, target, mask, amount=0.1):
    indices = np.where(mask != 0)
    points = np.hstack((indices[1][:, np.newaxis], indices[0][:, np.newaxis]))
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    amount = amount * np.max(extent)
    hull = cv2.convexHull(points)
    distances = np.zeros(points.shape[0])
    for index in range(points.shape[0]):
        point = int(points[index, 0]), int(points[index, 1])
        distances[index] = cv2.pointPolygonTest(hull, point, False)
    weights = np.clip(distances / amount, 0, 1)
    result = np.copy(target)
    result[indices[0], indices[1]] = (
        weights[:, np.newaxis] * source[indices[0], indices[1]]
        + (1 - weights[:, np.newaxis]) * target[indices[0], indices[1]]
    )
    new_mask = np.zeros_like(target).astype(np.float32)
    new_mask[indices[0], indices[1]] = weights[:, np.newaxis]
    return result, new_mask


def copy_fake_to_real(real, fake, mask, magnitude):
    mask = np.expand_dims(mask, 2)
    return fake * mask * magnitude + real * (1 - mask) + real * mask * (1 - magnitude)


def blend_fake_to_real(real, real_landmarks, fake, deformed_mask, kind, magnitude):
    real = ((real + 1) / 2 * 255).astype(np.uint8)
    fake = ((fake + 1) / 2 * 255).astype(np.uint8)
    target_mask = mask_postprocess(np.asarray(deformed_mask > 0, dtype=np.uint8))
    mode = ["rct-m", "rct-fs", "avg-align", "faceswap"][np.random.randint(4)]
    if mode != "faceswap":
        correction_mask = target_mask / 255.0
        correction_mask[correction_mask > 0] = 1
        if len(correction_mask.shape) < 3:
            correction_mask = np.expand_dims(correction_mask, 2)
        corrected = color_transfer(mode, fake, real, correction_mask)
    else:
        correction_mask = target_mask.copy()
        correction_mask[correction_mask > 0] = 255
        masked_real = cv2.bitwise_and(real, real, mask=correction_mask)
        masked_fake = cv2.bitwise_and(fake, fake, mask=correction_mask)
        corrected = faceswap_color(masked_real, masked_fake, np.array(real_landmarks))
    if target_mask.mean() < 0.005 or corrected.max() == 0:
        result = real
    elif kind == 0:
        result, _ = feather_blend(corrected, real, target_mask, 0.2 * np.random.rand())
    elif kind == 1:
        mask = (target_mask * 255).astype(np.uint8)
        left, top, width, height = cv2.boundingRect(mask)
        center = int(left + width / 2), int(top + height / 2)
        result = cv2.seamlessClone(corrected, real, mask, center, cv2.NORMAL_CLONE)
    else:
        result = copy_fake_to_real(real, corrected, target_mask, magnitude)
    return result, target_mask
