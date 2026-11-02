"""Strict original trunk and explicit-recognition initialization imports."""

from detectors._shared.checkpoint import read_weights


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    adapted, ignored = {}, []
    for key, value in weights.items():
        if "fc" in key:
            ignored.append(key)
            continue
        if "pointwise" in key and value.ndim == 2:
            value = value.unsqueeze(-1).unsqueeze(-1)
        adapted[key] = value
    expected = backbone.state_dict()
    counters = []
    for key, value in expected.items():
        if key.endswith(".num_batches_tracked") and key not in adapted:
            adapted[key] = value.clone()
            counters.append(key)
    missing, unexpected = set(expected) - set(adapted), set(adapted) - set(expected)
    allowed = {
        k
        for k in expected
        if k.startswith(("last_linear.", "adjust_channel."))
        and not k.endswith(".num_batches_tracked")
    }
    if missing != allowed or unexpected:
        raise ValueError(
            f"IID Xception initialization layout changed: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    backbone.load_state_dict(adapted, strict=False)
    return {
        **receipt,
        "matched_supplied_keys": len(adapted) - len(counters),
        "retained_fresh": sorted(missing),
        "ignored_fc": ignored,
        "initialized_counters": counters,
    }


def initialize_explicit(extractor, path):
    weights, receipt = read_weights(path)
    extractor.load_state_dict(weights, strict=True)
    return {**receipt, "strict_explicit_extractor_keys": len(weights)}
