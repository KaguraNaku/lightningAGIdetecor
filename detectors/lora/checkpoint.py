"""Legacy loralib import requires an explicit saved-weight mode."""

from detectors._shared.checkpoint import read_weights


def import_legacy(path, *, alpha, rank, weight_mode):
    if weight_mode not in ("merged", "unmerged"):
        raise ValueError("legacy_weight_mode must explicitly be merged or unmerged")
    weights, receipt = read_weights(path)
    weights = dict(weights)
    if weight_mode == "merged":
        # This inverse is subject to the rounding of the original in-place merge.
        for name, a in tuple(weights.items()):
            if name.endswith(".lora_A"):
                prefix = name.removesuffix("lora_A")
                b = weights[prefix + "lora_B"]
                weights[prefix + "weight"] = weights[prefix + "weight"] - (b @ a) * (
                    alpha / rank
                )
    if not any(k.endswith(".lora_A") for k in weights):
        raise ValueError("Legacy LoRA checkpoint has no adapter matrices")
    receipt.update(legacy_weight_mode=weight_mode, imported_weight_mode="unmerged")
    return weights, receipt
