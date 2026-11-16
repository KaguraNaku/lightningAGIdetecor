"""Lazy construction of detector-specific frozen vision adapters."""

from importlib import import_module

from backbones.load_backbone import load_backbone


ENCODERS = {
    "clip_vision_model": (
        "detectors.lightning_detetcor.encoders.clip",
        "CLIPReadoutEncoder",
    ),

}


def load_encoder(model_id, *, readout="global", **kwargs):
    """Load one backbone then its adapter, never all model families.

    Revision, offline, cache and dtype options pass through to load_backbone.
    readout="global_mean" returns the two views in one frozen forward pass.
    Use the same checkpoint's image processor separately; pixel preprocessing
    and resize/interpolation are not guessed or overridden by an encoder.
    """
    if readout not in ("global", "mean", "std", "mean_std", "global_mean"):
        raise ValueError("unknown vision readout")
    model = load_backbone(model_id, component="vision", **kwargs)
    family = model.config.model_type
    # Keep existing CLIP CLS checkpoints and the original public class intact.
    if family == "clip_vision_model" and readout == "global":
        module = import_module("detectors.lightning_detetcor.encoder")
        return module.CLIPVisionEncoder(model, model_id=str(model_id))
    if family not in ENCODERS:
        raise ValueError(f"Unsupported vision encoder: {family}")
    module_name, class_name = ENCODERS[family]
    cls = getattr(import_module(module_name), class_name)
    return cls(
        model, model_id=str(model_id), readout=readout, revision=kwargs.get("revision")
    )
