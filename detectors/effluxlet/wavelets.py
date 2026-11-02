"""Native Haar/zero separable analysis and synthesis, without optional libraries."""

from importlib.resources import files
import json

import torch
from torch import nn
from torch.nn import functional as F


def analysis(values, low, high, *, dim):
    channels = values.shape[1]
    shape = (1, 1, 2, 1) if dim == 2 else (1, 1, 1, 2)
    filters = torch.cat([low.reshape(shape), high.reshape(shape)] * channels, dim=0)
    if values.shape[dim] % 2:
        values = F.pad(values, (0, 0, 0, 1) if dim == 2 else (0, 1, 0, 0))
    return F.conv2d(
        values, filters, stride=(2, 1) if dim == 2 else (1, 2), groups=channels
    )


def synthesis(low_values, high_values, low, high, *, dim):
    channels = low_values.shape[1]
    shape = (1, 1, 2, 1) if dim == 2 else (1, 1, 1, 2)
    low = torch.cat([low.reshape(shape)] * channels, dim=0)
    high = torch.cat([high.reshape(shape)] * channels, dim=0)
    stride = (2, 1) if dim == 2 else (1, 2)
    return F.conv_transpose2d(
        low_values, low, stride=stride, groups=channels
    ) + F.conv_transpose2d(high_values, high, stride=stride, groups=channels)


class HaarAnalysis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, h0_row, h1_row, h0_col, h1_col):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = values.shape[-2:]
        bands = analysis(analysis(values, h0_row, h1_row, dim=3), h0_col, h1_col, dim=2)
        bands = bands.reshape(bands.shape[0], -1, 4, *bands.shape[-2:])
        return bands[:, :, 0].contiguous(), bands[:, :, 1:].contiguous()

    @staticmethod
    def backward(ctx, low, high):
        gradient = None
        if ctx.needs_input_grad[0]:
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(high, dim=2)
            lo = synthesis(low, lh, h0_col, h1_col, dim=2)
            hi = synthesis(hl, hh, h0_col, h1_col, dim=2)
            gradient = synthesis(lo, hi, h0_row, h1_row, dim=3)
            gradient = gradient[:, :, : ctx.shape[0], : ctx.shape[1]]
        return gradient, None, None, None, None


class HaarPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        profile = json.loads(
            files("configs").joinpath("migrations/effluxlet_haar.json").read_text()
        )
        low = torch.tensor(profile["dec_lo"][::-1], dtype=torch.get_default_dtype())
        high = torch.tensor(profile["dec_hi"][::-1], dtype=torch.get_default_dtype())
        self.register_buffer("h0_col", low.reshape(1, 1, 2, 1))
        self.register_buffer("h1_col", high.reshape(1, 1, 2, 1))
        self.register_buffer("h0_row", low.clone().reshape(1, 1, 1, 2))
        self.register_buffer("h1_row", high.clone().reshape(1, 1, 1, 2))

    def forward(self, images):
        low, scales, highs = images, [], []
        for _ in range(4):
            # Source passes col buffers in row argument slots; Haar coefficients
            # match and the filter-bank code reshapes them along each axis.
            low, high = HaarAnalysis.apply(
                low, self.h0_col, self.h1_col, self.h0_row, self.h1_row
            )
            scales.append(low)
            highs.append(high)
        return scales, highs
