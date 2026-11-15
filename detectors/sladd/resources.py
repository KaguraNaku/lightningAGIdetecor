"""Symbolic native-sized saved-graph inventory; not measured peak RAM/disk."""

from functools import lru_cache

import torch

from .initialization import _checkpoint_structure
from .losses import detector_losses, synthesizer_loss
from .sampling import policy_action


@lru_cache(maxsize=1)
def _native_saved_graph():
    with torch.device("meta"):
        model = _checkpoint_structure()
        persistent = {
            v.untyped_storage()._cdata for v in [*model.parameters(), *model.buffers()]
        }
        unique, held = {}, []

        def pack(value):
            storage = value.untyped_storage()
            if storage._cdata not in persistent:
                key = storage._cdata, value._version, str(value.dtype)
                unique[key] = storage.nbytes()
                held.append(storage)  # Prevent identity reuse during inventory.
            return value.detach()

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            policy_logits = model.synthesizer.netG(torch.empty(32, 6, 256, 256))
            region, kind, magnitude = [policy_action(value) for value in policy_logits]
            features, spatial = model.backbone.features(torch.empty(64, 3, 256, 256))
            cls, pooled = model.backbone.classifier(features)
            outputs = {
                "cls": cls,
                "prob": cls.softmax(1)[:, 1],
                "map": model.backbone.map(spatial),
                "type": model.backbone.type_fc(pooled),
                "mag": model.backbone.mag_fc(pooled),
                "label": torch.empty(64, dtype=torch.int64),
                "type_label": torch.empty(64, dtype=torch.int64),
                "mag_mask": torch.ones(64, dtype=torch.int64),
                "mag_label": magnitude[2].detach().repeat(2),
                "alt_mask": torch.empty(64, 1, 16, 16),
            }
            losses = detector_losses(outputs)
            policy, _ = synthesizer_loss(
                losses["overall"],
                log_prob=region[1] + kind[1] + magnitude[1],
                entropy=region[0] + kind[0] + magnitude[0],
            )
            assert losses["overall"].requires_grad and policy.requires_grad
        return {
            "policy_pairs": 32,
            "main_views": 64,
            "resolution": 256,
            "dtype": "float32",
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
            "logical_nonpersistent_saved_bytes": sum(unique.values()),
            "unique_storage_versions": len(unique),
            "scope": "Meta shape graph through both CNNs/policy sampling/objectives;pixel synthesis replaced only by its detached64-view boundary for sizing. Not numerical gradients or measured RAM/disk peak.",
        }


def native_saved_graph():
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("SLADD symbolic budget requires default float32")
    return dict(_native_saved_graph())
