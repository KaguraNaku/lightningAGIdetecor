import math

import torch
from torch import nn


def solve_ridge(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    *,
    block_size: int = 64,
    ridge_lambda: float = 1.0,
):
    """Solve a ridge system with layer-specific penalties.

    Parameters
    ----------
    gram : torch.Tensor of shape (n_features, n_features)
        Feature Gram matrix.
    rhs : torch.Tensor of shape (n_features,)
        Feature-target cross-product with centered targets.
    block_size : int, default=64
        Number of features per layer.
    ridge_lambda : float, default=1.0
        Positive multiplier for the layer-specific penalties.

    Returns
    -------
    weights : torch.Tensor of shape (n_features,)
        Fitted regression coefficients.
    """
    _validate_ridge_inputs(gram, rhs, block_size, ridge_lambda)
    penalty = _build_block_penalty(gram, block_size, ridge_lambda)
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
    block_size: int = 64,
    ridge_lambda: float = 1.0,
):
    """Fit ridge regression from features and labels.

    Parameters
    ----------
    features : torch.Tensor of shape (n_samples, n_features)
        Floating-point features, concatenated in layer order.
    labels : torch.Tensor of shape (n_samples,)
        Raw binary targets, converted to the feature dtype and device.
    block_size : int, default=64
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
    return solve_ridge(gram, rhs, block_size=block_size, ridge_lambda=ridge_lambda)


def _validate_config(dimension, block_size, ridge_lambda):
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if dimension % block_size:
        raise ValueError("dimension must be divisible by block_size")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")


def _validate_ridge_inputs(gram, rhs, block_size, ridge_lambda):
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not gram.shape[0]:
        raise ValueError("gram must be a nonempty square matrix")
    if rhs.shape != (gram.shape[0],):
        raise ValueError("rhs must contain one value per feature")
    _validate_config(gram.shape[0], block_size, ridge_lambda)
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise ValueError("gram and rhs must be finite")


def _build_block_penalty(gram, block_size, ridge_lambda):
    diagonal = torch.diag(gram)

    # group by layers
    blocks = diagonal.reshape(-1, block_size)
    layer_penalties = blocks.sum(dim=1) / block_size
    penalty = ridge_lambda * layer_penalties.repeat_interleave(block_size)

    if not torch.isfinite(penalty).all():
        raise ValueError("Regularization penalties must be finite")
    if not (penalty > 0).all():
        raise ValueError("Degenerate feature block")
    return penalty


def _validate_features(features, dimension=None, *, require_samples=False):
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if require_samples and not len(features):
        raise ValueError("features must contain at least one sample")
    if dimension is not None and features.shape[1] != dimension:
        raise ValueError(f"features must have {dimension} columns")
    if not features.is_floating_point():
        raise ValueError("features must be floating-point tensors")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")


def _validate_batch(features, labels, dimension=None):
    _validate_features(features, dimension, require_samples=True)
    if labels.shape != (len(features),):
        raise ValueError("labels must contain one value per sample")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only raw 0/1 targets")


def _validate_mapping_id(mapping_id, *, allow_none=False):
    if allow_none and mapping_id is None:
        return
    if not isinstance(mapping_id, str) or not mapping_id:
        raise ValueError("a frozen feature mapping fingerprint is required")


class RidgeClassifier(nn.Module):
    """Fixed-mapping ridge with additive sufficient statistics.

    Statistics and weights are buffers, saved and moved with the parent module.
    Fitting, addition and removal share the same solver and centered-target rule.
    """

    # Version 2 uses the base lambda directly, without sample-count scaling.
    VERSION = 2

    def __init__(self, config):
        super().__init__()
        self.dimension = config["dimension"]
        self.block_size = config["block_size"]
        self.ridge_lambda = float(config["ridge_lambda"])
        _validate_config(self.dimension, self.block_size, self.ridge_lambda)

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
            "version": self.VERSION,
            "dimension": self.dimension,
            "block_size": self.block_size,
            "ridge_lambda": self.ridge_lambda,
            "mapping_id": self.mapping_id,
        }

    def set_extra_state(self, state):
        self._validate_extra_state(state)
        self.mapping_id = state["mapping_id"]

    def _validate_extra_state(self, state):
        expected = self.get_extra_state()
        if not isinstance(state, dict) or not expected.keys() <= state.keys():
            raise ValueError("saved classifier metadata is incomplete")
        for name in ("version", "dimension", "block_size", "ridge_lambda"):
            if (
                type(state[name]) is not type(expected[name])
                or state[name] != expected[name]
            ):
                raise ValueError(
                    f"Saved classifier {name} does not match configuration"
                )
        _validate_mapping_id(state["mapping_id"], allow_none=not int(self.sample_count))
        if int(self.sample_count) < 0 or not 0 <= float(self.label_sum) <= int(
            self.sample_count
        ):
            raise ValueError("saved label/count statistics are inconsistent")

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
        _validate_batch(features, labels, self.dimension)
        return features.to(self.gram), labels.to(self.gram)

    def _check_mapping(self, mapping_id, *, replace=False):
        _validate_mapping_id(mapping_id)
        if (
            not replace
            and self.mapping_id is not None
            and mapping_id != self.mapping_id
        ):
            raise ValueError(
                "feature mapping changed; stored statistics cannot be reused"
            )

    def _solve(self, gram, feature_label_sum, feature_sum, label_sum, sample_count):
        # Center targets using the complete accumulated label mean.
        rhs = feature_label_sum - (label_sum / sample_count) * feature_sum

        # With a frozen feature mapping, repeating the full data k times scales
        # G, rhs and the block penalty R by k. With fixed lambda, (kG + kR) w = k rhs has the same solution.
        # This guarantees replication invariance, not invariance to new samples
        # with a different distribution. Extra sample-count scaling is disabled.
        # effective_lambda = self.ridge_lambda * max(
        #     1.0, (sample_count / self.dimension) ** 0.5
        # )
        return solve_ridge(
            gram, rhs, block_size=self.block_size, ridge_lambda=self.ridge_lambda
        )

    def _validate_removal(
        self, gram, feature_label_sum, feature_sum, label_sum, sample_count
    ):
        if sample_count < 0:
            raise ValueError("cannot remove more samples than the stored count")
        if not 0 <= float(label_sum) <= sample_count:
            raise ValueError("remaining binary label counts are inconsistent")
        if sample_count == 0:
            for remainder, previous in (
                (gram, self.gram),
                (feature_label_sum, self.feature_label_sum),
                (feature_sum, self.feature_sum),
            ):
                tolerance = 1e-10 * max(1.0, float(previous.abs().max()))
                if float(remainder.abs().max()) > tolerance:
                    raise ValueError("empty state has nonzero residual contributions")
            return
        # Necessary numerical checks, not proof of sample membership.
        symmetric = (gram + gram.T) / 2
        tolerance = 1e-10 * max(1.0, float(symmetric.norm()))
        if float(torch.linalg.eigvalsh(symmetric).min()) < -tolerance:
            raise ValueError("remaining Gram matrix is not positive semidefinite")

    @torch.no_grad()
    def _prepare_update(
        self, features, labels, *, mapping_id, direction=1, replace=False
    ):
        self._check_mapping(mapping_id, replace=replace)
        features, labels = self._prepare_batch(features, labels)
        old_count = 0 if replace else int(self.sample_count)

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

        # Solve a candidate state before changing stored statistics.
        if direction == -1:
            self._validate_removal(
                gram, feature_label_sum, feature_sum, label_sum, sample_count
            )
        if sample_count == 0:
            return None
        weights = self._solve(
            gram, feature_label_sum, feature_sum, label_sum, sample_count
        )
        return dict(
            gram=gram,
            feature_label_sum=feature_label_sum,
            feature_sum=feature_sum,
            label_sum=label_sum,
            sample_count=sample_count,
            weights=weights,
            mapping_id=mapping_id,
        )

    @torch.no_grad()
    def _commit_update(self, candidate):
        if candidate is None:
            return self.reset()
        for name in (
            "gram",
            "feature_label_sum",
            "feature_sum",
            "label_sum",
            "weights",
        ):
            getattr(self, name).copy_(candidate[name])
        self.sample_count.fill_(candidate["sample_count"])
        self.mapping_id = candidate["mapping_id"]
        return self

    def _update(self, features, labels, *, mapping_id, direction=1, replace=False):
        candidate = self._prepare_update(
            features,
            labels,
            mapping_id=mapping_id,
            direction=direction,
            replace=replace,
        )
        return self._commit_update(candidate)

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
        if mapping_id is not None:
            self._check_mapping(mapping_id)
        _validate_features(features, self.dimension)
        return features.to(self.weights) @ self.weights


class EqualWeightRidgeClassifier(nn.Module):
    """Two independent Ridge heads with raw, equal-weight score addition.

    No LDA, score standardization or joint branch fitting. Both candidate heads
    must solve successfully before an online update changes either head.
    """

    BRANCHES = ("global", "mean")

    def __init__(self, config):
        super().__init__()
        self.branches = nn.ModuleDict(
            {name: RidgeClassifier(config) for name in self.BRANCHES}
        )

    def _validate_views(self, features):
        if not isinstance(features, dict) or set(features) != set(self.BRANCHES):
            raise ValueError("features must contain exactly global and mean views")
        if len(features["global"]) != len(features["mean"]):
            raise ValueError("feature views must have the same sample count")

    def _update(self, features, labels, *, mapping_id, direction=1, replace=False):
        self._validate_views(features)
        if not isinstance(mapping_id, dict) or set(mapping_id) != set(self.BRANCHES):
            raise ValueError("a frozen mapping fingerprint is required for each view")
        candidates = {
            name: classifier._prepare_update(
                features[name],
                labels,
                mapping_id=mapping_id[name],
                direction=direction,
                replace=replace,
            )
            for name, classifier in self.branches.items()
        }
        for name, classifier in self.branches.items():
            classifier._commit_update(candidates[name])
        return self

    def fit(self, features, labels, *, mapping_id):
        return self._update(features, labels, mapping_id=mapping_id, replace=True)

    def partial_fit(self, features, labels, *, mapping_id):
        return self._update(features, labels, mapping_id=mapping_id)

    def remove_batch(self, features, labels, *, mapping_id):
        return self._update(features, labels, mapping_id=mapping_id, direction=-1)

    def branch_scores(self, features):
        self._validate_views(features)
        return {
            name: classifier(features[name])
            for name, classifier in self.branches.items()
        }

    def forward(self, features):
        scores = self.branch_scores(features)
        return scores["global"] + scores["mean"]
