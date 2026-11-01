"""Strictly translate torchvision ImageNet weights into the native trunk."""

from detectors._shared.checkpoint import read_weights


PREFIXES = {
    "conv1": "0",
    "bn1": "1",
    "layer1": "4",
    "layer2": "5",
    "layer3": "6",
    "layer4": "7",
}


def initialize_backbone(backbone, path):
    weights, receipt = read_weights(path)
    translated, ignored = {}, []
    for key, value in weights.items():
        if key in ("fc.weight", "fc.bias"):
            if tuple(value.shape) != ((1000, 512) if key == "fc.weight" else (1000,)):
                raise ValueError(
                    "ResNet34 ImageNet initialization needs the 1000-class source head"
                )
            ignored.append(key)
            continue
        root, separator, suffix = key.partition(".")
        if not separator or root not in PREFIXES:
            raise ValueError(f"Unexpected ResNet34 ImageNet key: {key}")
        translated[PREFIXES[root] + "." + suffix] = value
    counters = []
    for key, value in backbone.resnet.state_dict().items():
        if key.endswith(".num_batches_tracked") and key not in translated:
            translated[key] = value.clone()
            counters.append(key)
    backbone.resnet.load_state_dict(translated, strict=True)
    receipt.update(
        ignored_classifier_keys=ignored,
        initialized_batch_counters=counters,
        initialized_trunk_keys=sorted(translated),
    )
    return receipt
