"""Mapped immutable backbone storage with owned mutable residual/head weights."""

from detectors._shared.checkpoint import load_strict


def restore_weights(model, weights):
    load_strict(model, weights, assign=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.detach().clone()
