"""Native full-rank ZCA buffers and macro-only image scoring."""

import torch
from torch import nn


DIMENSION = 1024
PATCHES = 256
RIDGE = 0.01
LOGIT_SCALE = 0.001


class HybridZCAStatistics(nn.Module):
    def __init__(self, layers=(12, 18), *, memory_size=2048):
        super().__init__()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Native Hybrid ZCA requires default float32 tensors")
        if (
            not isinstance(layers, (tuple, list))
            or not layers
            or any(type(i) is not int or not 0 <= i < 24 for i in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError("Hybrid ZCA needs distinct zero-based CLIP-L layers")
        if type(memory_size) is not int or memory_size < 1:
            raise ValueError("Hybrid ZCA memory_size must be a positive integer")
        self.layers, self.memory_size = tuple(layers), memory_size
        for layer in self.layers:
            self.register_buffer(f"zca_mat_{layer}", torch.eye(DIMENSION))
            self.register_buffer(f"real_mean_{layer}", torch.zeros(DIMENSION))
            self.register_buffer(f"steering_vec_{layer}", torch.zeros(DIMENSION))
            # Retain source state layout even though these weights are unused
            # by its currently active classifier.
            self.register_buffer(f"weight_{layer}", torch.tensor(1.0))
        self.register_buffer(
            f"memory_bank_{self.layers[0]}", torch.zeros(memory_size, DIMENSION)
        )

    def validate_features(self, features, *, count=None):
        if set(features) != set(self.layers):
            raise ValueError("Hybrid ZCA features must cover every selected layer")
        device = getattr(self, f"real_mean_{self.layers[0]}").device
        sizes = set()
        for value in features.values():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[1] != DIMENSION
                or value.dtype != torch.float32
                or value.device != device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "Hybrid ZCA needs finite device-aligned N x1024 float32 features"
                )
            sizes.add(value.shape[0])
        if len(sizes) != 1 or (count is not None and sizes != {count}):
            raise ValueError("Hybrid ZCA layer sample counts differ")
        return next(iter(sizes))

    def score(self, features):
        count = self.validate_features(features)
        device = getattr(self, f"real_mean_{self.layers[0]}").device
        macro = torch.zeros(count, device=device)
        for layer in self.layers:
            feat = features[layer]
            real_mean = getattr(self, f"real_mean_{layer}")
            zca_mat = getattr(self, f"zca_mat_{layer}")
            steering_vec = getattr(self, f"steering_vec_{layer}")
            feat_white = (feat - real_mean) @ zca_mat
            layer_score = torch.matmul(feat_white, steering_vec)
            macro += layer_score
        return {
            "macro_score": macro,
            "micro_score": torch.zeros_like(macro),
            "total_score": macro,
        }

    @staticmethod
    def logits(total_score):
        return torch.stack(
            [-total_score * LOGIT_SCALE, total_score * LOGIT_SCALE], dim=1
        )

    @torch.no_grad()
    def fit_zca(self, features, labels):
        count = self.validate_features(features)
        if (
            labels.shape != (count,)
            or labels.device != next(iter(features.values())).device
            or not ((labels == 0) | (labels == 1)).all()
            or int((labels == 0).sum()) < 2
            or int((labels == 1).sum()) < 1
        ):
            raise ValueError("Hybrid ZCA requires at least2 real and1 fake samples")
        receipts = []
        for layer in self.layers:
            X = features[layer]
            X_real = X[labels == 0]
            X_fake = X[labels == 1]
            real_mean = X_real.mean(dim=0)
            getattr(self, f"real_mean_{layer}").copy_(real_mean)
            X_real_centered = X_real - real_mean
            cov = torch.matmul(X_real_centered.T, X_real_centered) / (
                X_real.shape[0] - 1
            )
            cov_reg = cov + RIDGE * torch.eye(DIMENSION, device=X.device)
            L, U = torch.linalg.eigh(cov_reg)
            if not torch.isfinite(L).all() or not (L > 0).all():
                raise ValueError(
                    "Hybrid ZCA regularized covariance is not positive finite"
                )
            inv_sqrt_L = torch.diag(1.0 / torch.sqrt(L))
            zca_matrix = U @ inv_sqrt_L @ U.T
            getattr(self, f"zca_mat_{layer}").copy_(zca_matrix)
            X_fake_white = (X_fake - real_mean) @ zca_matrix
            steering_vec = X_fake_white.mean(dim=0)
            if not torch.isfinite(steering_vec).all():
                raise ValueError("Hybrid ZCA steering vector is nonfinite")
            getattr(self, f"steering_vec_{layer}").copy_(steering_vec)
            receipts.append(
                {
                    "layer": layer,
                    "dimension": DIMENSION,
                    "ridge": RIDGE,
                    "min_eigenvalue": float(L.min()),
                    "max_eigenvalue": float(L.max()),
                }
            )
        return receipts
