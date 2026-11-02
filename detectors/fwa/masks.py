"""Native five hull choices, piecewise deformation and random mask blur."""

import cv2
import numpy as np


def hull_mask(image, points, kind):
    if kind == "normal-hull":
        landmarks = np.array(points)
    elif kind == "inner-hull":
        landmarks = np.array(points[17:])
    elif kind == "inner-hull-no-eyebrow":
        landmarks = np.array(points[27:])
    elif kind == "mouth-hull":
        landmarks = np.concatenate([points[2:15], points[29].reshape(1, -1)], axis=0)
    elif kind == "whole-hull":
        height = points[9][1] - points[22][1]
        landmarks = []
        for index in range(27):
            point = points[index]
            if 5 <= index <= 11:
                point = [point[0], max(0, point[1] + 15)]
            if 18 <= index <= 27:
                point = [point[0], max(0, point[1] - height // 4)]
            landmarks.append(point)
        landmarks = np.array(landmarks, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported native FWA hull: {kind}")
    hull = cv2.convexHull(landmarks).astype(int)
    if kind == "whole-hull":
        hull = np.reshape(hull, (1, -1, 2))
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [hull], (255, 255, 255))
    return mask


def deform_mask(mask, standard_deviation):
    from skimage.transform import PiecewiseAffineTransform, warp

    height, width = mask.shape[:2]
    rows = np.linspace(0, height, 4).astype(np.int32)
    columns = np.linspace(0, width, 4).astype(np.int32)
    rows, columns = np.meshgrid(rows, columns)
    anchors = np.vstack([rows.flat, columns.flat]).T
    deformed = anchors + np.random.normal(0, standard_deviation, size=anchors.shape)
    np.clip(deformed[:, 0], 0, height - 1, deformed[:, 0])
    np.clip(deformed[:, 1], 0, width - 1, deformed[:, 1])
    transform = PiecewiseAffineTransform()
    transform.estimate(anchors.astype(np.float32), deformed.astype(np.float32))
    return (warp(mask, transform) * 255).astype(np.uint8)


def face_mask(points, image, std=20, deform=True):
    choices = [
        "normal-hull",
        "inner-hull",
        "inner-hull-no-eyebrow",
        "mouth-hull",
        "whole-hull",
    ]
    maximum = hull_mask(image, points, "whole-hull")
    if deform:
        kind = choices[np.random.randint(len(choices))]
        mask = hull_mask(image, points, kind)
        if np.random.rand() < 0.9:
            mask = deform_mask(mask, std)
    else:
        mask = maximum.copy()
    mask = mask * (maximum // 255)
    if deform and np.random.rand() < 0.9:
        kernel = 2 * np.random.randint(1, 10) - 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask[:, :, 0]
