"""Load the entire native B4 ImageNet state, retaining stem and1000-way head."""

from detectors._shared.checkpoint import read_weights


def initialize_b4(encoder, path):
    weights, receipt = read_weights(path)
    expected, counters = encoder.model.state_dict(), []
    weights = dict(weights)
    for key, value in expected.items():
        if key.endswith(".num_batches_tracked") and key not in weights:
            weights[key] = value.clone()
            counters.append(key)
    encoder.model.load_state_dict(weights, strict=True)
    return {
        **receipt,
        "strict_original_B4_stem_and1000head": True,
        "initialized_counters": counters,
        "fresh_projection": "1792_to512_Conv1x1",
    }
