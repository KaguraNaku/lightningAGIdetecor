"""Strict complete two-network SLADD weights; no optimizer/RNG resume claim."""

from functools import lru_cache

import torch

from detectors._shared.checkpoint import load_strict, read_weights
from .initialization import _checkpoint_structure


def selected_recipe():
    # These source branches are not encoded by parameter shapes alone.
    return {
        "backbone_id": "xception_sladd",
        "backbone_mode": "original",
        "resolution": 256,
        "in_channels": 3,
        "num_classes": 2,
        "backbone_dropout": False,
        "policy_in_channels": 6,
        "policy_regions": 10,
        "policy_types": 4,
        "policy_magnitudes": 1,
        "policy_dropout": 0.5,
        "detector_type_classes": 5,
        "type_margin": 0.5,
        "type_scale": 30,
        "map_loss_weight": 0.1,
        "type_loss_weight": 0.05,
        "magnitude_loss_weight": 0.1,
        "magnitude_loss_layout": "native_pairwise_broadcast",
        "magnitude_action": "continuous_sigmoid",
        "policy_reward": "detached_scalar_overall",
        "entropy_weight": 1e-5,
        "score_type": "probability",
    }


def require_recipe(recipe):
    expected = selected_recipe()
    if (
        not isinstance(recipe, dict)
        or set(recipe) != set(expected)
        or any(
            type(recipe[key]) is not type(value) or recipe[key] != value
            for key, value in expected.items()
        )
    ):
        raise ValueError(
            "SLADD weights require the explicit selected native two-network recipe"
        )
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Selected native SLADD weights require default float32")


@lru_cache(maxsize=1)
def _schema():
    with torch.device("meta"):
        model = _checkpoint_structure()
    assert len(model.state_dict()) == 562 and len(list(model.parameters())) == 322
    assert len(list(model.detector_parameters())) == 162
    assert len(list(model.synthesizer.parameters())) == 160
    return {
        key: (tuple(value.shape), value.dtype, tuple(value.stride()))
        for key, value in model.state_dict().items()
    }


def detector_schema(*, recipe):
    require_recipe(recipe)  # Check before cache lookup, including default dtype.
    return dict(_schema())


def validate_detector_state(weights, *, recipe):
    schema = detector_schema(recipe=recipe)
    if not isinstance(weights, dict) or set(weights) != set(schema):
        raise ValueError(
            "SLADD requires all562 exact main/policy keys, including every auxiliary head and BN buffer"
        )
    for key, (shape, dtype, stride) in schema.items():
        value = weights[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or tuple(value.stride()) != stride
        ):
            raise ValueError(
                f"Invalid SLADD native CPU state shape, dtype or stride: {key}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite SLADD state: {key}")
    return weights


def detector_weights(path, *, recipe):
    require_recipe(recipe)
    weights, receipt = read_weights(path)
    validate_detector_state(weights, recipe=recipe)
    return weights, {
        **receipt,
        "recipe": selected_recipe(),
        "layout": "complete_native_SLADD_Detector562",
        "operation": "strict_full_two_network_weights_import;training_status_unknown",
        "dtype_policy": "Exact native CPU dtypes/strides;no casts,head substitution or missing-buffer reconstruction",
        "backbone_pretraining_translation": False,
        "pretrained_asset_required_for_restore": False,
        "optimizer_scheduler_epoch_rng_resumed": False,
        "policy_optimizer_state_restored": False,
    }


def restore_detector(weights, *, recipe):
    validate_detector_state(weights, recipe=recipe)
    # Native state loading copies values into independent parameter/buffer
    # allocations. Preserve that behavior even for aliased caller mappings.
    with torch.no_grad():
        independent = {
            key: value.detach().clone(memory_format=torch.contiguous_format)
            for key, value in weights.items()
        }
    with torch.device("meta"):
        model = _checkpoint_structure()
    load_strict(model, independent, assign=True)
    main, policy = (
        list(model.detector_parameters()),
        list(model.synthesizer.parameters()),
    )
    if (
        len(main) != 162
        or len(policy) != 160
        or len(list(model.parameters())) != 322
        or {id(p) for p in main} & {id(p) for p in policy}
        or any(not p.requires_grad for p in model.parameters())
    ):
        raise ValueError(
            "SLADD restoration must retain all322 trainable parameters and native162/160 optimizer partition"
        )
    model.initialization = {
        "operation": "complete_checkpoint_restore",
        "ImageNet_file_read": False,
        "states": 562,
    }
    model.synthesizer.netG.initialization = {
        "operation": "complete_checkpoint_restore",
        "ImageNet_file_read": False,
    }
    return model
