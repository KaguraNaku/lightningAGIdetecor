"""Native detached AGDA with separable reflect-padded Gaussian filtering."""

import random

import torch
from torch import nn
from torch.nn import functional as F


class GaussianBlur(nn.Module):
    def forward(self, images):
        sigma = images.new_tensor([[7.0, 7.0]])
        kernels = []
        for index in (1, 0):
            std = sigma[:, index].view(1, 1)
            mean = images.new_tensor([[5.0]])
            position = (
                torch.arange(11, device=images.device, dtype=images.dtype) - mean
            ).expand(1, -1)
            kernel = torch.exp(-position.pow(2.0) / (2 * std.pow(2.0)))
            kernels.append(kernel / kernel.sum(-1, keepdim=True))
        channels = images.shape[1]
        horizontal = (
            kernels[0][:, None, None, :]
            .expand(-1, channels, -1, -1)
            .reshape(-1, 1, 1, 11)
        )
        vertical = (
            kernels[1][:, None, :, None]
            .expand(-1, channels, -1, -1)
            .reshape(-1, 1, 11, 1)
        )
        images = F.conv2d(
            F.pad(images, (5, 5, 0, 0), mode="reflect"), horizontal, groups=channels
        )
        return F.conv2d(
            F.pad(images, (0, 0, 5, 5), mode="reflect"), vertical, groups=channels
        )


class AGDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = GaussianBlur()
        self.last_augmented = None
        self.last_indices = None

    @torch.no_grad()
    def agda(self, images, attentions):
        weights = torch.sqrt(torch.sum(attentions, dim=(2, 3)) + 1)
        attentions = F.interpolate(
            attentions, images.shape[-2:], mode="bilinear", align_corners=True
        )
        indices = torch.distributions.categorical.Categorical(weights).sample()
        gather = indices.view(-1, 1, 1, 1).repeat(1, 1, *images.shape[-2:])
        attention = torch.gather(attentions, 1, gather)
        maximum = torch.max(attention.view(attention.shape[0], 1, -1), 2)[0] + 1e-8
        attention = attention / maximum.view(attention.shape[0], 1, 1, 1)
        threshold, zoom = random.uniform(0.4, 0.6), random.uniform(3, 5)
        bottom = torch.sigmoid((torch.tensor(0.0) - threshold) * zoom)
        attention = (torch.sigmoid((attention - threshold) * zoom) - bottom) / (
            1 - bottom
        )
        blurred = F.interpolate(
            images, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        blurred = self.filter(blurred)
        blurred += torch.randn_like(blurred) * 0.1
        blurred = F.interpolate(
            blurred, images.shape[-2:], mode="bilinear", align_corners=True
        )
        augmented = images * (1 - attention) + blurred * attention
        self.last_augmented, self.last_indices = augmented, indices
        return augmented, indices
