"""Literal native RGB-gray high-pass IDCT, without clipping or renormalization."""

import cv2
import numpy as np
import torch


def normalized_rgb(pixels):
    return (
        torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        .float()
        .div(255)
        .sub(0.5)
        .div(0.5)
    )


def frequency(pixels):
    gray = np.float32(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY))
    coefficients = cv2.dct(gray)
    rows, columns = coefficients.shape
    x, y = np.ogrid[:rows, :columns]
    high = np.ones((rows, columns), dtype=np.uint8)
    high[(columns / rows * x + y) < 0.33 * columns] = 0
    restored = cv2.idct(high * coefficients)
    # The active native call has r1=.33,r2=.33; only high mask is consumed.
    return torch.from_numpy(np.ascontiguousarray(((restored / 255 - 0.5) / 0.5)[None]))


def frequency_from_normalized(images):
    """Recover exact native uint8 pixels for the ordinary predict(Tensor) API.

    The batch evaluator uses the explicit IDCT from its raw-pixel collator.
    Arbitrary continuous/half-rounded RGB tensors need an explicit IDCT input;
    silently quantizing them would define a different frequency representation.
    """
    if (
        images.ndim != 4
        or images.shape[1:] != (3, 256, 256)
        or not torch.isfinite(images).all()
    ):
        raise ValueError("LRL predict requires native normalized RGB256 tensors")
    source = images.detach().float().cpu()
    decoded = ((source * 0.5 + 0.5) * 255).round()
    if (
        decoded.min() < 0
        or decoded.max() > 255
        or not torch.equal(decoded.div(255).sub(0.5).div(0.5), source)
    ):
        raise ValueError(
            "LRL continuous RGB tensors require explicit aligned IDCT via predict_batch"
        )
    pixels = decoded.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    return torch.stack([frequency(image) for image in pixels])
