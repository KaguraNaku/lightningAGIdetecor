import math

import torch
from torch import nn


def solve_ridge(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    *,
    width: int = 64,
    ridge_lambda: float = 1.0,
):
    """Solve a ridge system with layer-specific penalties.

    Parameters
    ----------
    gram : torch.Tensor of shape (n_features, n_features)
        Feature Gram matrix.
    rhs : torch.Tensor of shape (n_features,)
        Feature-target cross-product with centered targets.
    width : int, default=64
        Number of features per layer.
    ridge_lambda : float, default=1.0
        Positive multiplier for the layer-specific penalties.

    Returns
    -------
    weights : torch.Tensor of shape (n_features,)
        Fitted regression coefficients.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not gram.shape[0]:
        raise ValueError("gram must be a nonempty square matrix")
    if rhs.shape != (gram.shape[0],):
        raise ValueError("rhs must contain one value per feature")
    if not isinstance(width, int) or width < 1 or gram.shape[0] % width:
        raise ValueError("width must be positive and divide n_features")

    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise ValueError("gram and rhs must be finite")

    diagonal = torch.diag(gram)

    # group by layers
    blocks = diagonal.reshape(-1, width)
    layer_penalties = blocks.sum(dim=1) / width
    penalty = ridge_lambda * layer_penalties.repeat_interleave(width)

    if not torch.isfinite(penalty).all():
        raise ValueError("Regularization penalties must be finite")

    if not (penalty > 0).all():
        raise ValueError("Degenerate feature block")

    system_matrix = gram + torch.diag(penalty)
    weights = torch.linalg.solve(system_matrix, rhs)
    if not torch.isfinite(weights).all():
        raise ValueError("Fitted coefficients must be finite")
    return weights


@torch.no_grad()
def fit_ridge(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    width: int = 64,
    ridge_lambda: float = 1.0,
):
    """Fit ridge regression from features and labels.

    Parameters
    ----------
    features : torch.Tensor of shape (n_samples, n_features)
        Floating-point features, concatenated in layer order.
    labels : torch.Tensor of shape (n_samples,)
        Floating-point targets with the same dtype and device as features.
    width : int, default=64
        Number of features per layer.
    ridge_lambda : float, default=1.0
        Positive multiplier for the layer-specific penalties.

    Returns
    -------
    weights : torch.Tensor of shape (n_features,)
        Fitted regression coefficients.
    """
    _validate_batch(features, labels)
    labels = labels.to(features)
    gram = features.T @ features
    targets = labels - labels.mean()
    rhs = features.T @ targets

    return solve_ridge(
        gram,
        rhs,
        width=width,
        ridge_lambda=ridge_lambda,
    )


def _validate_batch(features, labels):
    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("features must be a nonempty 2D matrix")
    if not features.is_floating_point():
        raise ValueError("features must be floating-point tensors")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    if labels.shape != (len(features),):
        raise ValueError("labels must contain one value per sample")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only raw 0/1 targets")


class RidgeClassifier(nn.Module):
    """Fixed-mapping ridge with additive sufficient statistics.

    Statistics and weights are buffers, saved and moved with the parent module.
    Fitting, addition and removal share the same solver and centered-target rule.
    """

    def __init__(self, config):
        super().__init__()
        self.dimension = config["dimension"]
        self.block_size = config["block_size"]
        self.ridge_lambda = float(config["ridge_lambda"])
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not isinstance(self.dimension, int) or self.dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if self.dimension % self.block_size:
            raise ValueError("dimension must be divisible by block_size")
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be finite and positive")

        self.mapping_id = None
        self.register_buffer(
            "gram", torch.zeros(self.dimension, self.dimension, dtype=torch.float64)
        )
        self.register_buffer(
            "feature_label_sum", torch.zeros(self.dimension, dtype=torch.float64)
        )
        self.register_buffer(
            "feature_sum", torch.zeros(self.dimension, dtype=torch.float64)
        )
        self.register_buffer("label_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            "weights", torch.zeros(self.dimension, dtype=torch.float64)
        )

    def get_extra_state(self):
        return {
            "version": 1,
            "dimension": self.dimension,
            "block_size": self.block_size,
            "ridge_lambda": self.ridge_lambda,
            "mapping_id": self.mapping_id,
        }

    def set_extra_state(self, state):
        for name in ("version", "dimension", "block_size", "ridge_lambda"):
            if state[name] != self.get_extra_state()[name]:
                raise ValueError(
                    f"Saved classifier {name} does not match configuration"
                )
        self.mapping_id = state["mapping_id"]

    @torch.no_grad()
    def reset(self):
        """Clear accumulated statistics and mark the classifier as unfitted.

        @gram: feature Gram matrix, features.T @ features.
        @feature_label_sum: feature-label products, features.T @ labels.
        @feature_sum: feature sums, features.sum(dim=0).
        @label_sum: label sum; fake sample count for raw 0/1 targets.
        @sample_count: total number of accumulated samples.
        @weights: ridge coefficients; unusable until sample_count > 0.
        """
        for buffer in self.buffers():
            buffer.zero_()
        self.mapping_id = None
        return self

    def _prepare_batch(self, features, labels):
        _validate_batch(features, labels)
        if features.shape[1] != self.dimension:
            raise ValueError(f"features must have {self.dimension} columns")
        return features.to(self.gram), labels.to(self.gram)

    def _solve(self, gram, feature_label_sum, feature_sum, label_sum, sample_count):
        # Center targets using the complete accumulated label mean.
        rhs = feature_label_sum - (label_sum / sample_count) * feature_sum
        effective_lambda = self.ridge_lambda * max(
            1.0, (sample_count / self.dimension) ** 0.5
        )
        return solve_ridge(
            gram, rhs, width=self.block_size, ridge_lambda=effective_lambda
        )

    @torch.no_grad()
    def _update(self, features, labels, *, mapping_id, direction=1, replace=False):
        if not isinstance(mapping_id, str) or not mapping_id:
            raise ValueError("a frozen feature mapping fingerprint is required")
        if (
            not replace
            and self.mapping_id is not None
            and mapping_id != self.mapping_id
        ):
            raise ValueError(
                "feature mapping changed; stored statistics cannot be reused"
            )
        features, labels = self._prepare_batch(features, labels)
        old_count = 0 if replace else int(self.sample_count)
        if direction == -1 and len(labels) > old_count:
            raise ValueError("cannot remove more samples than the stored count")

        gram = direction * (features.T @ features)
        feature_label_sum = direction * (features.T @ labels)
        feature_sum = direction * features.sum(dim=0)
        label_sum = direction * labels.sum()
        if not replace:
            gram = self.gram + gram
            feature_label_sum = self.feature_label_sum + feature_label_sum
            feature_sum = self.feature_sum + feature_sum
            label_sum = self.label_sum + label_sum
        sample_count = old_count + direction * len(labels)
        if not 0 <= float(label_sum) <= sample_count:
            raise ValueError("remaining binary label counts are inconsistent")

        # Solve a candidate state before changing stored statistics.
        if sample_count == 0:
            for remainder, previous in (
                (gram, self.gram),
                (feature_label_sum, self.feature_label_sum),
                (feature_sum, self.feature_sum),
            ):
                tolerance = 1e-10 * max(1.0, float(previous.abs().max()))
                if float(remainder.abs().max()) > tolerance:
                    raise ValueError("empty state has nonzero residual contributions")
            return self.reset()
        if direction == -1:
            symmetric = (gram + gram.T) / 2
            tolerance = 1e-10 * max(1.0, float(symmetric.norm()))
            if float(torch.linalg.eigvalsh(symmetric).min()) < -tolerance:
                raise ValueError("remaining Gram matrix is not positive semidefinite")
        weights = self._solve(
            gram, feature_label_sum, feature_sum, label_sum, sample_count
        )
        self.gram.copy_(gram)
        self.feature_label_sum.copy_(feature_label_sum)
        self.feature_sum.copy_(feature_sum)
        self.label_sum.copy_(label_sum)
        self.sample_count.fill_(sample_count)
        self.weights.copy_(weights)
        self.mapping_id = mapping_id
        return self

    def fit(self, features, labels, *, mapping_id):
        """Replace the stored statistics and fit a new classifier."""
        return self._update(features, labels, mapping_id=mapping_id, replace=True)

    def partial_fit(self, features, labels, *, mapping_id):
        """Add samples in the same frozen mapping and refit analytically."""
        return self._update(features, labels, mapping_id=mapping_id)

    def remove_batch(self, features, labels, *, mapping_id):
        """Subtract supplied contributions; the caller must establish membership."""
        return self._update(features, labels, mapping_id=mapping_id, direction=-1)

    def forward(self, features, *, mapping_id=None):
        if not int(self.sample_count):
            raise ValueError("fit before prediction")
        if mapping_id is not None and mapping_id != self.mapping_id:
            raise ValueError("feature mapping mismatch")
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise ValueError(
                f"features must be a 2D matrix with {self.dimension} columns"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        return features.to(self.weights) @ self.weights
