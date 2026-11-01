"""ImageNet body import and native RGB-quarter replication into twelve channels."""

import torch
from torch import nn

from detectors._shared.checkpoint import read_weights


def replace_stem(backbone, rgb_weights=None):
    backbone.conv1 = nn.Conv2d(12, 32, 3, stride=2, bias=False)
    if rgb_weights is not None:
        with torch.no_grad():
            for band in range(4):
                backbone.conv1.weight[:, band * 3 : (band + 1) * 3].copy_(
                    rgb_weights / 4.0
                )


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    stem = weights.get("conv1.weight")
    if stem is None or tuple(stem.shape) != (32, 3, 3, 3):
        raise ValueError("F3Net initialization requires the native RGB ImageNet stem")
    adapted, ignored = {}, []
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
        raise ValueError("F3Net ImageNet trunk keys differ from the declared layout")
    result = backbone.load_state_dict(adapted, strict=False)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("F3Net initialization key accounting mismatch")
    replace_stem(backbone, stem)
    receipt.update(
        ignored_classifier_keys=ignored,
        initialized_batch_counters=counters,
        initialized_keys=sorted(adapted),
        stem_conversion="four_RGB_copies_each_scaled_by_one_quarter",
    )
    return receipt
