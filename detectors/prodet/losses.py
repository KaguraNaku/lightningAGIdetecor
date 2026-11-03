# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from DeepfakeBench soft_ce_loss.py;remove registry/base-framework coupling.
"""Literal native soft confidence-to-target andlog-softmax objective."""

import torch
from torch import nn
from torch.nn import functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def target_generation(self, inputs, confidence):
        class_num = inputs.size(-1)
        if class_num == 3:
            right_idx = torch.ceil(confidence).long()
            left_idx = torch.floor(confidence).long()
            confidence_new = torch.where(confidence < 1, confidence, confidence - 1)
            target = torch.zeros_like(inputs)
            target[torch.arange(inputs.size(0)), right_idx] = confidence_new
            target[torch.arange(inputs.size(0)), left_idx] = torch.where(
                target[torch.arange(inputs.size(0)), left_idx].long() == 1,
                float(1),
                (1 - confidence_new).double(),
            ).float()
        elif class_num == 2:
            target = torch.stack((1 - confidence, confidence), dim=1)
        else:
            raise ValueError(f"{class_num} class soft ce-loss is not implemented yet")
        return target

    def forward(self, inputs, confidence):
        """
        Computes the cross-entropy loss with soft labels.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.
            confidence: A float representing the confidence for the target class.

        Returns:
            A scalar tensor representing the custom cross-entropy loss.
        """
        soft_targets = self.target_generation(inputs, confidence.float())
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss
