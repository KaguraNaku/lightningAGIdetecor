"""Import the native flat I3D/Kinetics state and select its first head row."""

from detectors._shared.checkpoint import read_weights


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path, encoding="latin1")
    supplied = {}
    for key, value in weights.items():
        target = key.replace("resnet.", "")
        if target in supplied:
            raise ValueError(
                f"Ambiguous I3D checkpoint key after prefix removal: {target}"
            )
        supplied[target] = value
    head_weight = supplied.get("head.projection.weight")
    head_bias = supplied.get("head.projection.bias")
    if (
        head_weight is None
        or head_bias is None
        or head_weight.ndim != 2
        or head_weight.shape[0] < 1
        or head_weight.shape[1] != 2048
        or head_bias.shape != (head_weight.shape[0],)
    ):
        raise ValueError(
            "Native I3D needs a nonempty, aligned2048-input projection head"
        )
    source_classes = head_weight.shape[0]
    supplied["head.projection.weight"] = head_weight[:1, :]
    supplied["head.projection.bias"] = head_bias[:1]
    backbone.load_state_dict(supplied, strict=True)
    return {
        **receipt,
        "native_pickle_encoding": "latin1",
        "source_projection_rows": source_classes,
        "selected_projection_rows": [0],
        "matched_keys": len(supplied),
        "prefix_rule": "literal resnet. replacement, as in native constructor",
        "collision_policy": "reject ambiguous translated keys",
    }
