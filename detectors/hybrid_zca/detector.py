"""Frozen CLIP-L feature extraction composed with native ZCA statistics."""

import torch
from torch import nn

from .statistics import DIMENSION, PATCHES, HybridZCAStatistics


class HybridZCADetector(nn.Module):
    def __init__(self, backbone, *, layers=(12, 18), memory_size=2048):
        super().__init__()
        config = backbone.config
        if (
            config.hidden_size != DIMENSION
            or config.num_hidden_layers != 24
            or config.num_attention_heads != 16
            or config.image_size != 224
            or config.patch_size != 14
        ):
            raise ValueError("Hybrid ZCA requires the native CLIP ViT-L/14 geometry")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.statistics = HybridZCAStatistics(layers, memory_size=memory_size)
        self.backbone.eval()

    @property
    def vision_model(self):
        # Transformers4 wraps the vision transformer; Transformers5 exposes it
        # directly. Keep one registered backbone and the stock state layout.
        return getattr(self.backbone, "vision_model", self.backbone)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def features(self, images, *, collect_real_patches=False):
        if (
            images.ndim != 4
            or images.shape[1:] != (3, 224, 224)
            or images.dtype != torch.float32
            or not torch.isfinite(images).all()
        ):
            raise ValueError("Hybrid ZCA requires finite N x3 x224 x224 float32 pixels")
        if torch.is_autocast_enabled(images.device.type):
            raise ValueError("Hybrid ZCA feature extraction requires float32 precision")
        features, patches, handles = {}, None, []
        primary = self.statistics.layers[0]

        def capture(layer):
            def hook(module, inputs, output):
                nonlocal patches
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[1:] != (PATCHES + 1, DIMENSION):
                    raise ValueError("Unexpected CLIP-L token geometry")
                # Reduce the original strided slice, exactly as in the source;
                # cloning the patch tensor first changes reduction layout.
                features[layer] = hidden[:, 1:, :].mean(dim=1).clone().detach()
                if collect_real_patches and layer == primary:
                    patches = hidden[:, 1:, :].clone().detach()

            return hook

        self.backbone.eval()
        try:
            for layer in self.statistics.layers:
                handles.append(
                    self.vision_model.encoder.layers[layer].register_forward_hook(
                        capture(layer)
                    )
                )
            self.vision_model(images)
        finally:
            for handle in handles:
                handle.remove()
        self.statistics.validate_features(features, count=images.shape[0])
        return features, patches

    def forward_outputs(self, images):
        features, _ = self.features(images)
        scores = self.statistics.score(features)
        logits = self.statistics.logits(scores["total_score"])
        return {
            "cls": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "feat": features[self.statistics.layers[0]],
            **scores,
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]
