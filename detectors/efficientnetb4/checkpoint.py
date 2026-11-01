"""ImageNet initialization preserves the native freshly replaced stem."""

from detectors._shared.checkpoint import read_weights


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    expected = backbone.efficientnet.state_dict()
    excluded = {
        "_conv_stem.weight": (48, 3, 3, 3),
        "_fc.weight": (1000, 1792),
        "_fc.bias": (1000,),
    }
    ignored, adapted = [], {}
    for key, value in weights.items():
        if key in excluded:
            if tuple(value.shape) != excluded[key]:
                raise ValueError(
                    f"Unexpected EfficientNet-B4 ImageNet tensor shape: {key}"
                )
            ignored.append(key)
        else:
            adapted[key] = value
    counters = []
    for key, value in expected.items():
        if key.endswith(".num_batches_tracked") and key not in adapted:
            adapted[key] = value.clone()
            counters.append(key)
    if set(adapted) != set(expected) - {"_conv_stem.weight"}:
        raise ValueError(
            "EfficientNet-B4 ImageNet trunk keys differ from the declared layout"
        )
    result = backbone.efficientnet.load_state_dict(adapted, strict=False)
    if result.missing_keys != ["_conv_stem.weight"] or result.unexpected_keys:
        raise ValueError("EfficientNet-B4 initialization key accounting mismatch")
    receipt.update(
        ignored_native_replaced_keys=ignored,
        initialized_batch_counters=counters,
        initialized_trunk_keys=sorted(adapted),
        stem="retained_native_fresh_unpadded_convolution",
    )
    return receipt
