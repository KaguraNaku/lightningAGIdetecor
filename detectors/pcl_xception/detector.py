"""Native self-correlation, including unused phi and the raw first target column."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def masks4d(masks):
    value = masks.squeeze(1)
    result = 1 - torch.abs(value[:, :, :, None, None] - value[:, None, None, :, :])
    # Native loop starts EVERY outer row with real_mask, not affinity.
    result[:, :, 0] = value[:, None, :, :]
    return result


class Correlation(nn.Module):
    def __init__(self, channels=728):
        super().__init__()
        self.in_channels = channels
        self.sig = nn.Sigmoid()
        self.theta = nn.Conv2d(channels, channels, 1)
        self.phi = nn.Conv2d(channels, channels, 1)

    def forward(self, features):
        batch = features.size(0)
        left = self.theta(features).view(batch, self.in_channels, -1)
        right = self.theta(features).view(batch, self.in_channels, -1)
        scores = torch.matmul(left.permute(0, 2, 1), right) / math.sqrt(
            self.in_channels
        )
        scores = self.sig(scores.permute(0, 2, 1).contiguous())
        return scores.view(batch, *features.size()[2:], *features.size()[2:])


class PCLDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.PCL = Correlation(728)
        self.mask_down_sampling = nn.UpsamplingBilinear2d(scale_factor=16 / 256)

    def forward_outputs(self, images, *, masks=None, inference=False):
        features, spatial = self.backbone.features(images)
        if inference:
            predicted, target = None, None
        else:
            if masks is None:
                raise ValueError("PCL training requires synchronized source masks")
            predicted = self.PCL(spatial)
            target = masks4d(self.mask_down_sampling(masks))
        logits, _ = self.backbone.classifier(features)
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": features,
            "pcl_map": predicted,
            "pcl_gt_map": target,
        }

    def forward(self, images):
        return self.forward_outputs(images, inference=True)["cls"]

    def loss_terms(self, outputs, labels):
        correlation = (
            F.binary_cross_entropy(outputs["pcl_map"], outputs["pcl_gt_map"])
            if outputs["pcl_map"] is not None
            else 0
        )
        classification = F.cross_entropy(outputs["cls"], labels)
        return {
            "overall": classification + correlation,
            "pcl_loss": correlation,
            "det_loss": classification,
        }
