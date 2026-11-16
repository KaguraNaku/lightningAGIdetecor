"""Source-domain calibration and fixed-mapping classifier updates."""

import hashlib

import torch

from .detector import LightningDetector
from .mapping import FeatureMapping, GlobalMeanMapping
from .ridge import EqualWeightRidgeClassifier, RidgeClassifier
from .search_layers import search_layers


def mapping_id(detector):
    """Fingerprint each frozen mapping, including its encoder and view identity."""
    identity = getattr(detector.backbone, "feature_id", None)
    if identity is None:
        identity = getattr(
            detector.backbone, "model_id", type(detector.backbone).__qualname__
        )
    if isinstance(detector.mapping, GlobalMeanMapping):
        return {
            name: _mapping_id(f"{identity}/{name}", mapping)
            for name, mapping in detector.mapping.branches.items()
        }
    return _mapping_id(identity, detector.mapping)


def _mapping_id(identity, mapping):
    fingerprint = hashlib.sha256(identity.encode())
    for tensor in (
        mapping.selected_layers,
        mapping.means,
        mapping.scales,
        mapping.basis,
    ):
        value = tensor.detach().cpu().contiguous()
        fingerprint.update(str((tuple(value.shape), value.dtype)).encode())
        fingerprint.update(value.view(torch.uint8).numpy().tobytes())
    return fingerprint.hexdigest()


def _make_classifier(mapping, config):
    return RidgeClassifier(
        {
            "dimension": len(mapping.selected_layers) * mapping.output_dim,
            "block_size": mapping.output_dim,
            "ridge_lambda": config.get("ridge_lambda", 0.15),
        }
    ).to(device=mapping.means.device)


def _make_mapping(config, device):
    settings = {
        name: config[name]
        for name in ("layers_count", "input_dim", "output_dim", "eps")
    }
    return FeatureMapping(settings).to(device=device)


def build_detector(backbone, config, *, selected_layers):
    """Construct modules for explicit layers, without searching or fitting.

    Used to restore a checkpoint with its saved layer selection. ``topk`` is
    unnecessary here: the actual selected layers determine classifier width.
    """
    mapping = _make_mapping(config, next(backbone.parameters()).device)
    mapping.select_layers(selected_layers)
    return LightningDetector(backbone, mapping, _make_classifier(mapping, config))


def load_detector(backbone, config, state_dict):
    """Restore a fresh detector and verify its mapping fingerprint before use.

    Pass a freshly constructed encoder; loading restores its weights as well.
    Earlier mask-based/version-1 checkpoints require explicit migration.
    """
    if "mapping.selected_layers" not in state_dict:
        raise ValueError("checkpoint must contain mapping.selected_layers")
    layers = state_dict["mapping.selected_layers"]
    detector = build_detector(backbone, config, selected_layers=layers)
    expected_layers = detector.mapping.selected_layers.clone()
    detector.load_state_dict(state_dict)
    if not torch.equal(detector.mapping.selected_layers, expected_layers):
        raise ValueError("saved layer indices must be in ascending order")
    if (
        int(detector.classifier.sample_count)
        and mapping_id(detector) != detector.classifier.mapping_id
    ):
        raise ValueError(
            "saved classifier fingerprint does not match the restored mapping"
        )
    return detector


@torch.no_grad()
def calibrate(backbone, features, labels, config):
    """Return a new fitted detector using source-training features and labels.

    Parameters
    ----------
    backbone : torch.nn.Module
        Feature encoder whose forward returns (N, layers, channels).
    features : torch.Tensor of shape (N, layers, channels)
        Source-training features produced by the same encoder.
    labels : torch.Tensor of shape (N,)
        Source-training raw targets: real=0, fake=1. Both classes are required.
    config : dict
        Mapping settings, topk and search/final ridge regularization settings.
        Optional selected_layers bypasses search; indices are zero-based.

    Returns
    -------
    LightningDetector
        New fitted modules. Existing detectors are not modified on failure.
    """
    if labels.shape != (len(features),):
        raise ValueError("labels must contain one value per sample")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only raw 0/1 targets")
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("calibration requires both real and fake samples")
    topk = config.get("topk", 8)
    if not isinstance(topk, int) or not 1 <= topk <= config["layers_count"]:
        raise ValueError("topk must be between 1 and layers_count")

    mapping = _make_mapping(config, features.device)
    mapping.fit(features, labels)
    projected = mapping(features)
    layers = config.get("selected_layers")
    if layers is None:
        layers, _ = search_layers(
            features=projected.to(dtype=torch.float64),
            labels=labels.to(device=projected.device, dtype=torch.float64),
            block_size=mapping.output_dim,
            topk=topk,
            ridge_lambda=config.get("search_ridge_lambda", 1.0),
        )
    # Explicit layers bypass search and use the mapping's index validation.
    mapping.select_layers(layers)
    layers = mapping.selected_layers.tolist()
    blocks = projected.reshape(len(projected), mapping.layers_count, mapping.output_dim)
    selected = blocks[:, layers, :].flatten(start_dim=1)
    detector = LightningDetector(backbone, mapping, _make_classifier(mapping, config))
    detector.classifier.fit(selected, labels, mapping_id=mapping_id(detector))
    return detector


@torch.no_grad()
def partial_fit(detector, features, labels):
    """Add samples while keeping layer selection and mapping fixed."""
    detector.classifier.partial_fit(
        detector.mapping(features), labels, mapping_id=mapping_id(detector)
    )
    return detector


@torch.no_grad()
def remove_batch(detector, features, labels):
    """Remove known fitted records while keeping layer selection and mapping fixed."""
    detector.classifier.remove_batch(
        detector.mapping(features), labels, mapping_id=mapping_id(detector)
    )
    return detector


def evenly_spaced_layers(layers_count, count=8):
    """Zero-based indices, including both endpoints; no source/target ranking.

    For 24 blocks and eight retained layers: [0, 3, 7, 10, 13, 16, 20, 23].
    Rounding uses ties-to-even, matching the established NumPy protocol.
    """
    if (
        not isinstance(layers_count, int)
        or isinstance(layers_count, bool)
        or layers_count < 1
    ):
        raise ValueError("layers_count must be a positive integer")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= layers_count
    ):
        raise ValueError("layer count must be between 1 and layers_count")
    return (
        torch.linspace(0, layers_count - 1, count, dtype=torch.float64).round().long()
    )


def build_global_mean_detector(backbone, config, *, selected_layers=None, device=None):
    """Build the fixed-layer Global + Mean pipeline without fitting.

    The encoder must use readout="global_mean". For CLIP this means normalized
    hidden CLS tokens and normalized patch means, not the final projected image
    embedding. Other adapters retain their native definition of global pooling.
    Defaults: eight evenly spaced layers, rank 128, real variance epsilon 1e-6,
    uniform mean-loss Ridge lambda 0.15. Set layers_count and input_dim explicitly.
    ``device`` controls mapping/solver buffers independently of the backbone.
    """
    if getattr(backbone, "readout", None) != "global_mean":
        raise ValueError('the backbone must use readout="global_mean"')
    settings = {
        "layers_count": config["layers_count"],
        "input_dim": config["input_dim"],
        "output_dim": config.get("output_dim", 128),
        "eps": config.get("eps", 1e-6),
    }
    if selected_layers is None:
        selected_layers = evenly_spaced_layers(
            settings["layers_count"], config.get("topk", 8)
        )
    if device is None:
        device = next(backbone.parameters()).device
    mapping = GlobalMeanMapping(settings, selected_layers=selected_layers).to(
        device=device
    )
    dimension = len(mapping.branches["global"].selected_layers) * settings["output_dim"]
    classifier = EqualWeightRidgeClassifier(
        {
            "dimension": dimension,
            "block_size": settings["output_dim"],
            "ridge_lambda": config.get("ridge_lambda", 0.15),
        }
    ).to(device=device)
    return LightningDetector(backbone, mapping, classifier)


@torch.no_grad()
def calibrate_global_mean(backbone, features, labels, config):
    """Fit a new fixed-layer, independent-head detector from source features.

    features is {"global": (N,L,D), "mean": (N,L,D)} from the same frozen encoder.
    Both raw classes (real=0, fake=1) are required. Real samples set mean/scale;
    both classes set the uncentered second moment and supervised Ridge heads.
    No validation labels, LDA, layer search or gradient updates are used.
    """
    if not isinstance(features, dict) or set(features) != {"global", "mean"}:
        raise ValueError("features must contain exactly global and mean views")
    if labels.shape != (len(features["global"]),):
        raise ValueError("labels must contain one value per sample")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only raw 0/1 targets")
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("calibration requires both real and fake samples")
    detector = build_global_mean_detector(
        backbone, config, device=features["global"].device
    )
    detector.mapping.fit(features, labels)
    detector.classifier.fit(
        detector.mapping(features), labels, mapping_id=mapping_id(detector)
    )
    return detector


def load_global_mean_detector(backbone, config, state_dict, *, device=None):
    """Restore both branches and validate layer identities and online statistics."""
    key = "mapping.branches.global.selected_layers"
    if key not in state_dict:
        raise ValueError("checkpoint must contain global/mean selected layers")
    detector = build_global_mean_detector(
        backbone, config, selected_layers=state_dict[key], device=device
    )
    expected = detector.mapping.branches["global"].selected_layers.clone()
    detector.load_state_dict(state_dict)
    ids = mapping_id(detector)
    for name, mapping in detector.mapping.branches.items():
        if not torch.equal(mapping.selected_layers, expected):
            raise ValueError(
                "saved branches must use identical ascending layer indices"
            )
        classifier = detector.classifier.branches[name]
        if int(classifier.sample_count) and classifier.mapping_id != ids[name]:
            raise ValueError(
                "saved classifier fingerprint does not match the restored mapping"
            )
    heads = detector.classifier.branches
    if int(heads["global"].sample_count) != int(heads["mean"].sample_count) or float(
        heads["global"].label_sum
    ) != float(heads["mean"].label_sum):
        raise ValueError("saved branches must contain the same labeled samples")
    return detector
