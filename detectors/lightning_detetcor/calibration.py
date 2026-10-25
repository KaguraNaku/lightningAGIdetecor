"""Source-domain calibration and fixed-mapping classifier updates."""

import hashlib

import torch

from .detector import LightningDetector
from .mapping import FeatureMapping
from .ridge import RidgeClassifier
from .search_layers import search_layers


def mapping_id(detector):
    """Fingerprint the encoder identity, layer selection and fitted mapping."""
    identity = getattr(
        detector.backbone, "model_id", type(detector.backbone).__qualname__
    )
    fingerprint = hashlib.sha256(identity.encode())
    mapping = detector.mapping
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
    layers, _ = search_layers(
        features=projected.to(dtype=torch.float64),
        labels=labels.to(device=projected.device, dtype=torch.float64),
        width=mapping.output_dim,
        topk=topk,
        ridge_lambda=config.get("search_ridge_lambda", 1.0),
    )
    blocks = projected.reshape(len(projected), mapping.layers_count, mapping.output_dim)
    selected = blocks[:, layers, :].flatten(start_dim=1)
    mapping.select_layers(layers)
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
