"""SPSL ImageNet import averages RGB stem weights and repeats them four times."""

import torch
from torch import nn

from detectors._shared.checkpoint import read_weights


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    adapted, ignored = {}, []
    stem = weights.get("conv1.weight")
    if stem is None or tuple(stem.shape) != (32, 3, 3, 3):
        raise ValueError(
            "SPSL initialization requires the native 3-channel ImageNet stem"
        )
    for key, value in weights.items():
        if key == "conv1.weight":
            continue
        if key.startswith(("fc.", "last_linear.")):
            ignored.append(key)
            continue
        if "pointwise" in key and value.ndim == 2:
            value = value[:, :, None, None]
        adapted[key] = value
    expected = backbone.state_dict()
    counters = []
    for key, value in expected.items():
        if key.endswith(".num_batches_tracked") and key not in adapted:
            adapted[key] = value.clone()
            counters.append(key)
    missing, unexpected = set(expected) - set(adapted), set(adapted) - set(expected)
    allowed = {
        key
        for key in expected
        if key == "conv1.weight" or key.startswith(("last_linear.", "adjust_channel."))
    }
    if unexpected or missing - allowed:
        raise ValueError("SPSL ImageNet trunk keys differ from the declared layout")
    result = backbone.load_state_dict(adapted, strict=False)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("SPSL initialization key accounting mismatch")
    # Preserve the native replacement constructor's RNG draws as well as its
    # exact mean-and-repeat rule, including the absence of any extra scaling.
    backbone.conv1 = nn.Conv2d(4, 32, 3, stride=2, bias=False)
    with torch.no_grad():
        backbone.conv1.weight.copy_(stem.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1))
    receipt.update(
        ignored_classifier_keys=ignored,
        initialized_batch_counters=counters,
        initialized_keys=sorted(adapted),
        stem_conversion="RGB_mean_repeated_four_times_without_rescaling",
    )
    return receipt
