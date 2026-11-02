"""Assign complete snapshots without duplicating immutable SVD factor storage."""

from torch import nn
from detectors._shared.checkpoint import load_strict


def restore_weights(model, weights):
    load_strict(model, weights, assign=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.detach().clone()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name)
                if value is not None:
                    setattr(module, name, value.clone())
