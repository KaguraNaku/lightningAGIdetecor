"""Native ImageNet initialization with fresh classifier/type/magnitude/map heads."""

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
        key
        for key in expected
        if key.startswith(("last_linear.", "type_fc.", "mag_fc.", "map."))
    }
    if unexpected or missing != allowed:
        raise ValueError(
            f"PCL Xception pretrain layout changed: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    result = backbone.load_state_dict(adapted, strict=False)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("PCL native initialization accounting changed")
    return {
        **receipt,
        "ignored_fc": ignored,
        "retained_fresh": sorted(missing),
        "initialized_counters": counters,
        "matched_supplied_keys": len(adapted) - len(counters),
    }
