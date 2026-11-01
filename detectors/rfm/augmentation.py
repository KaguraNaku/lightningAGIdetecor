"""Native FAM, three ranked rectangles and normalized uniform replacement noise."""

import random

import numpy as np
import torch


class RFMAugmenter:
    def __init__(self, backbone):
        self.backbone = backbone
        self.last_fam = self.last_mask = self.last_augmented = None

    def cal_fam(self, images):
        self.backbone.zero_grad()
        inputs = images.detach().clone().requires_grad_()
        features = self.backbone.features(inputs)
        # The legacy backbone returns (logits, classifier-activated features).
        # Calling classifier preserves its in-place ReLU, even though its logits
        # are not the native FAM target. Do not replace this with class logits.
        _ = self.backbone.classifier(features)
        target = features[:, 1] - features[:, 0]
        target.backward(torch.ones(target.shape, device=target.device))
        return torch.max(torch.abs(inputs.grad), dim=1, keepdim=True)[0]

    def apply(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 256, 256):
            raise ValueError(
                "Native RFM augmentation requires RGB batches at 256 pixels"
            )
        self.backbone.eval()
        fam = self.cal_fam(images)
        mask = torch.ones_like(fam)
        for index in range(len(fam)):
            ranked = np.argsort(fam[index].cpu().numpy().flatten())[::-1]
            points = 0
            for position in ranked:
                row, column = divmod(int(position), 256)
                if mask[index, 0, row, column] == 1:
                    height, width = random.randint(1, 120), random.randint(1, 120)
                    up, left = random.randint(1, height), random.randint(1, width)
                    top, bottom = max(row - up, 0), min(row + height - up, 256)
                    start, end = max(column - left, 0), min(column + width - left, 256)
                    mask[index, :, top:bottom, start:end] = 0
                    points += 1
                    if points >= 3:
                        break
        augmented = mask * images + (1 - mask) * (torch.rand_like(images) * 2 - 1)
        self.backbone.train()
        self.last_fam, self.last_mask, self.last_augmented = fam, mask, augmented
        return augmented
