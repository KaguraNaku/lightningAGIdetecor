"""Strict native ImageNet trunk import; retain fresh adjustment and unused heads."""

from detectors._shared.checkpoint import read_weights


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    supplied, ignored = {}, []
    for key, value in weights.items():
        if "fc" in key:
            ignored.append(key)
            continue
        if "pointwise" in key and value.ndim == 2:
            value = value.unsqueeze(-1).unsqueeze(-1)
        supplied[key] = value
    expected, counters = backbone.state_dict(), []
    for key, value in expected.items():
        if key.endswith(".num_batches_tracked") and key not in supplied:
            supplied[key] = value.clone()
            counters.append(key)
    missing = set(expected) - set(supplied)
    unexpected = set(supplied) - set(expected)
    allowed = {
        key
        for key in expected
        if key.startswith(("last_linear.", "adjust_channel."))
        and not key.endswith(".num_batches_tracked")
    }
    if missing != allowed or unexpected:
        raise ValueError(
            f"UCF Xception initialization layout changed: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    backbone.load_state_dict(supplied, strict=False)
    return {
        **receipt,
        "matched_supplied_keys": len(supplied) - len(counters),
        "retained_fresh": sorted(missing),
        "ignored_fc": ignored,
        "initialized_counters": counters,
    }
