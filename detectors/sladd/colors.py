"""Only the four color correction paths selected by native SLADD synthesis."""

import cv2
import numpy as np


def scale_channel(value):
    minimum, maximum = value.min(), value.max()
    bounds = max([minimum, 0]), min([maximum, 255])
    if minimum < bounds[0] or maximum > bounds[1]:
        return (bounds[1] - bounds[0]) * (value - minimum) / (
            maximum - minimum
        ) + bounds[0]
    return value


def masked_reinhard(target, source, mask):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_stats = [(v.mean(), v.std()) for v in cv2.split(source * mask)]
    target_stats = [(v.mean(), v.std()) for v in cv2.split(target * mask)]
    channels = []
    for value, (source_mean, source_std), (target_mean, target_std) in zip(
        cv2.split(target), source_stats, target_stats, strict=True
    ):
        value -= target_mean
        value = (source_std / target_std) * value
        value += source_mean
        channels.append(scale_channel(value))
    return cv2.cvtColor(cv2.merge(channels).astype(np.uint8), cv2.COLOR_LAB2BGR)


def masked_mean(source, target, mask):
    result = np.copy(source)
    indices = np.where(mask != 0)
    selected_target = target[indices[0], indices[1]].astype(np.int32)
    selected_source = source[indices[0], indices[1]].astype(np.int32)
    shifted = selected_source - np.mean(selected_source, axis=0)
    shifted = shifted + np.mean(selected_target, axis=0)
    result[indices[0], indices[1]] = np.clip(shifted, 0, 255)
    return result


def average_align(source, target, mask):
    current, old = source.copy() * mask, target.copy() * mask
    for channel in range(current.shape[2]):
        delta = old[:, :, channel].mean() - current[:, :, channel].mean()
        # Preserve source scalar/clamp/dtype ordering rather than choosing a
        # vectorized expression with different NumPy scalar promotion.
        for row in range(current.shape[0]):
            for column in range(current.shape[1]):
                value = current[row, column, channel] + delta
                value = max(0.0, value)
                value = min(1.0, value)
                current[row, column, channel] = value
    return current


def color_transfer(mode, source, target, mask):
    source = source.astype(dtype=np.float32) / 255.0
    target = target.astype(dtype=np.float32) / 255.0
    if mode in {"rct-m", "rct-fs"}:
        source = np.clip(source * 255, 0, 255).astype(np.uint8)
        target = np.clip(target * 255, 0, 255).astype(np.uint8)
        value = (
            masked_reinhard(source, target, mask)
            if mode == "rct-m"
            else masked_mean(source, target, mask)
        )
        value = np.clip(value.astype(np.float32) / 255.0, 0.0, 1.0)
    elif mode == "avg-align":
        value = np.clip(average_align(source, target, mask), 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported SLADD color mode: {mode}")
    return np.clip(value * 255, 0, 255).astype(np.uint8)


def faceswap_color(target, source, landmarks):
    width = 0.75 * np.linalg.norm(
        np.mean(landmarks[list(range(42, 48))], axis=0)
        - np.mean(landmarks[list(range(36, 42))], axis=0)
    )
    width = int(width)
    if width % 2 == 0:
        width += 1
    target_blur = cv2.GaussianBlur(target, (width, width), 0)
    source_blur = cv2.GaussianBlur(source, (width, width), 0).astype(int)
    source_blur += 128 * (source_blur <= 1)
    result = (
        source.astype(np.float64)
        * target_blur.astype(np.float64)
        / source_blur.astype(np.float64)
    )
    return np.clip(result, 0, 255).astype(np.uint8)
