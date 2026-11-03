"""Separate native pre-conversion I3D initialization from trained FTCN import."""

from detectors._shared.checkpoint import read_weights


def initialization_weights(path):
    weights, origin = read_weights(path, encoding="latin1")
    supplied = {}
    for name, value in weights.items():
        target = name.replace("resnet.", "")
        if target in supplied:
            raise ValueError(f"Ambiguous FTCN initialization key: {target}")
        supplied[target] = value
    head = supplied.get("head.projection.weight")
    bias = supplied.get("head.projection.bias")
    if (
        head is None or bias is None or head.ndim != 2
        or head.shape[0] < 1 or head.shape[1] != 2048
        or bias.shape != (head.shape[0],)
    ):
        raise ValueError("FTCN init_weights needs the pre-conversion I3D projection head")
    supplied["head.projection.weight"] = head[:1, :]
    supplied["head.projection.bias"] = bias[:1]
    return supplied, {
        **origin,
        "source_projection_rows": head.shape[0],
        "selected_projection_rows": [0],
        "native_pickle_encoding": "latin1",
        "prefix_rule": "literal resnet. replacement; reject collisions",
        "load_order": "full I3D initialization, strict copy, fresh temporal conversion, discard s5, new transformer head",
        "spatial_filter_transfer": "replacement convolutions are freshly initialized, as in native FTCN",
    }


def trained_weights(path):
    weights, origin = read_weights(path, encoding="latin1")
    if all(name.startswith("resnet.") for name in weights):
        layout = "native_detector"
    elif (
        not any(name.startswith("resnet.") for name in weights)
        and "head.time_T.mlp_head.1.weight" in weights
    ):
        weights = {f"resnet.{name}": value for name, value in weights.items()}
        layout = "native_backbone"
    else:
        raise ValueError("FTCN checkpoint needs an unambiguous detector/backbone layout")
    head = weights.get("resnet.head.time_T.mlp_head.1.weight")
    bias = weights.get("resnet.head.time_T.mlp_head.1.bias")
    if head is None or bias is None or head.shape != (1, 1024) or bias.shape != (1,):
        raise ValueError("FTCN trained weights need the stop5 transformer head; use init_weights for I3D initialization")
    weights = {
        name: value.float() if value.is_floating_point() else value
        for name, value in weights.items()
    }
    return weights, {
        **origin,
        "layout": layout,
        "native_pickle_encoding": "latin1",
        "floating_import_dtype": "float32_as_native_constructor",
    }
