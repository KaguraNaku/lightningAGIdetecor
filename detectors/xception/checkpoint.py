"""Explicit Xception ImageNet initialization and strict detector import."""

from detectors._shared.checkpoint import read_weights


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    adapted = {}
    ignored = []
    for key, value in weights.items():
        if key.startswith(("fc.", "last_linear.")):
            ignored.append(key)
            continue
        if "pointwise" in key and value.ndim == 2:
            value = value[:, :, None, None]
        adapted[key] = value
    reference = backbone.state_dict()
    initialized_counters = []
    for key, value in reference.items():
        if key.endswith(".num_batches_tracked") and key not in adapted:
            adapted[key] = value.clone()
            initialized_counters.append(key)
    missing = set(reference) - set(adapted)
    unexpected = set(adapted) - set(reference)
    allowed = {
        key for key in reference if key.startswith(("last_linear.", "adjust_channel."))
    }
    if unexpected or missing - allowed:
        raise ValueError(
            f"Unexpected Xception pretrain layout: missing={sorted(missing - allowed)}, unexpected={sorted(unexpected)}"
        )
    # Strictly validate every supplied tensor; only declared classifier/adapter
    # initialization omissions are permitted for ImageNet backbone import.
    result = backbone.load_state_dict(adapted, strict=False)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("Xception pretrain key accounting mismatch")
    receipt.update(
        ignored_classifier_keys=ignored,
        initialized_keys=sorted(adapted),
        initialized_batch_counters=initialized_counters,
        missing_initialization_keys=sorted(missing),
    )
    return receipt
