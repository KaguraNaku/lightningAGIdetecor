# SPDX-License-Identifier: Apache-2.0
# Selected AM-Softmax arithmetic adapted from Intel's am_softmax.py (2018).
"""Selected native SLADD objectives, including the pairwise magnitude broadcast."""

import torch
from torch.nn import functional as F


def type_loss(logits, targets):
    """Native AMSoftmaxLoss defaults: cos margin=.5,scale30,gamma0,t1.

    Inputs are the raw five-class type head,without cosine normalization.
    Bool mask has the same selected-branch values as the source uint8 mask;
    unsupported arc/focal/hard-example modes are not silently substituted.
    """
    index = torch.zeros_like(logits, dtype=torch.bool)
    index.scatter_(1, targets.data.view(-1, 1), True)
    output = torch.where(index, logits - 0.5, logits)
    return F.cross_entropy(30 * output, targets)


def detector_losses(outputs):
    labels, logits = outputs["label"], outputs["cls"]
    if "map" in outputs:
        loss_type = type_loss(outputs["type"], outputs["type_label"]).mean()
        # Deliberate native behavior: [N,1]*[N] produces [N,N],not [N,1].
        # The MSE target remains [N] and broadcasts over rows. No unsqueeze.
        loss_mag = F.mse_loss(
            outputs["mag"] * outputs["mag_mask"],
            outputs["mag_label"] * outputs["mag_mask"],
        ).mean()
        loss_maps = F.mse_loss(outputs["map"], outputs["alt_mask"])
    else:
        loss_type, loss_mag, loss_maps = 0, 0, 0
    classification = F.cross_entropy(logits, labels)
    overall = classification + 0.1 * loss_maps + 0.05 * loss_type + 0.1 * loss_mag
    return {
        "overall": overall,
        "loss_type": loss_type,
        "loss_mag": loss_mag,
        "loss_maps": loss_maps,
    }


def synthesizer_loss(overall, *, log_prob, entropy):
    if log_prob is None:
        return 0, 0
    # The misleading source name normlized_lm is an unchanged scalar detach.
    reward = overall.detach()
    score_loss = torch.mean(-log_prob * reward)
    entropy_penalty = torch.mean(entropy)
    return score_loss - 1e-5 * entropy_penalty, entropy_penalty
