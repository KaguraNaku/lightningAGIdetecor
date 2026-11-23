"""Spectral mapping, Ridge fitting, and equal-weight global and patch-mean scores."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import scipy.linalg as la

from . import config

BRANCHES = ("cls", "mean")


def positive_ridge_lambda(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("ridge_lambda must be finite and positive")
    return float(value)


@dataclass
class SufficientStatistics:
    """Additive statistics for one Ridge readout.

    The rows of Z are projected features. The labels y are zero for real
    samples and one for fake samples.

    Attributes
    ----------
    gram : numpy.ndarray
        Feature Gram matrix Z.T @ Z.
    feature_label_sum : numpy.ndarray
        Label-weighted feature sum Z.T @ y, summing fake samples.
    feature_sum : numpy.ndarray
        Sum of all projected features, Z.sum(axis=0).
    label_sum : float
        Sum of the labels, y.sum(), equal to the number of fake samples.
    sample_count : int
        Number of rows n.
    """

    gram: np.ndarray
    feature_label_sum: np.ndarray
    feature_sum: np.ndarray
    label_sum: float
    sample_count: int

    @classmethod
    def empty(cls, width):
        return cls(np.zeros((width, width)), np.zeros(width), np.zeros(width), 0.0, 0)

    @classmethod
    def from_rows(cls, z, y):
        z, y = np.asarray(z, dtype=np.float64), np.asarray(y)
        if (
            z.ndim != 2
            or y.shape != (len(z),)
            or (y.dtype != np.bool_ and not np.isin(y, [0, 1]).all())
        ):
            raise ValueError("Aligned binary-labelled projected rows required")
        y = y.astype(np.float64, copy=False)
        return cls(
            gram=z.T @ z,
            feature_label_sum=z.T @ y,
            feature_sum=z.sum(0),
            label_sum=float(y.sum()),
            sample_count=len(y),
        )

    @classmethod
    def from_state(cls, state):
        return cls(
            gram=np.asarray(state["gram"], dtype=np.float64),
            feature_label_sum=np.asarray(state["feature_label_sum"], dtype=np.float64),
            feature_sum=np.asarray(state["feature_sum"], dtype=np.float64),
            label_sum=float(state["label_sum"]),
            sample_count=int(state["sample_count"]),
        )

    def state_dict(self):
        return {
            "gram": self.gram,
            "feature_label_sum": self.feature_label_sum,
            "feature_sum": self.feature_sum,
            "label_sum": np.asarray(self.label_sum),
            "sample_count": np.asarray(self.sample_count, dtype=np.int64),
        }

    def add(self, other):
        if self.gram.shape != other.gram.shape:
            raise ValueError(
                "Cannot combine statistics under different feature dimensions"
            )
        return type(self)(
            gram=self.gram + other.gram,
            feature_label_sum=self.feature_label_sum + other.feature_label_sum,
            feature_sum=self.feature_sum + other.feature_sum,
            label_sum=self.label_sum + other.label_sum,
            sample_count=self.sample_count + other.sample_count,
        )

    def solve(self, *, ridge_lambda=config.RIDGE_LAMBDA):
        if not 0 < self.label_sum < self.sample_count:
            raise ValueError("Ridge fitting requires both classes")
        ridge_lambda = positive_ridge_lambda(ridge_lambda)
        label_mean = self.label_sum / self.sample_count
        rhs = self.feature_label_sum - label_mean * self.feature_sum
        # Minimize ||Z w - (y - mean(y))||² / n + ridge_lambda * ||w||².
        penalty = self.sample_count * ridge_lambda
        system = self.gram.copy()
        system.flat[:: len(system) + 1] += penalty
        if not np.isfinite(rhs).all():
            raise ValueError("Ridge right-hand side must be finite")
        factor = la.cho_factor(system, lower=True)
        w = la.cho_solve(factor, rhs, check_finite=False)
        residual = float(
            la.norm(self.gram @ w + penalty * w - rhs) / max(la.norm(rhs), 1e-30)
        )
        if not math.isfinite(residual) or residual >= 1e-8 or not np.isfinite(w).all():
            raise ValueError("Analytic ridge normal-equation check failed")
        return w, {
            "relative_normal_equation_residual": residual,
            "n": self.sample_count,
            "n_fake": int(self.label_sum),
            "label_mean": label_mean,
            "ridge_lambda": ridge_lambda,
            "system_diagonal_penalty": penalty,
        }


@dataclass
class SpectralMapping:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray

    def __post_init__(self):
        self.mean, self.scale, self.basis = [
            np.asarray(x, dtype=np.float64) for x in (self.mean, self.scale, self.basis)
        ]
        if (
            self.mean.ndim != 2
            or self.scale.shape != self.mean.shape
            or self.basis.ndim != 3
            or self.basis.shape[:2] != self.mean.shape
            or not all(
                np.isfinite(x).all() for x in (self.mean, self.scale, self.basis)
            )
            or not (self.scale > 0).all()
        ):
            raise ValueError("Invalid frozen spectral mapping")

    @property
    def rank(self):
        return self.basis.shape[-1]

    def project(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != self.mean.shape:
            raise ValueError("Raw readouts differ from declared mapping")
        return np.einsum(
            "nld,ldr->nlr", (x - self.mean) / self.scale, self.basis, optimize=True
        ).reshape(len(x), -1)

    def fold(self, weights):
        weights = np.asarray(weights).reshape(self.mean.shape[0], self.rank)
        beta = np.einsum("ldr,lr->ld", self.basis, weights) / self.scale
        return beta, -float(np.sum(beta * self.mean))

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale, "basis": self.basis}


class MappingAccumulator:
    """Raw moments for a streaming, two-pass source mapping fit."""

    def __init__(self, shape=None):
        self.n = self.n_real = 0
        self.sum = self.real_sum = self.real_square = self.gram = None
        if shape is not None:
            self._allocate(shape)

    def _allocate(self, shape):
        layers, width = shape
        self.sum = np.zeros(shape)
        self.real_sum = np.zeros(shape)
        self.real_square = np.zeros(shape)
        self.gram = np.zeros((layers, width, width))

    def update(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y)
        if (
            x.ndim != 3
            or y.shape != (len(x),)
            or (y.dtype != np.bool_ and not np.isin(y, [0, 1]).all())
        ):
            raise ValueError("Aligned binary readouts required")
        if self.sum is None:
            self._allocate(x.shape[1:])
        if x.shape[1:] != self.sum.shape:
            raise ValueError("Readout shape changed during mapping fit")
        real = x[y == 0]
        self.n += len(x)
        self.n_real += len(real)
        self.sum += x.sum(0)
        self.real_sum += real.sum(0)
        self.real_square += np.square(real).sum(0)
        self.gram += np.einsum("nld,nle->lde", x, x, optimize=True)

    def fit_mapping(self, rank=config.PROJECTION_RANK):
        if not self.n_real or self.n < rank or not 0 < rank <= self.sum.shape[-1]:
            raise ValueError("Insufficient source rows or genuine references")
        mean = self.real_sum / self.n_real
        scale = np.sqrt(np.maximum(self.real_square / self.n_real - mean**2, 0) + 1e-6)
        bases = []
        for layer in range(len(mean)):
            mu, sx = mean[layer], self.sum[layer]
            gram = (
                self.gram[layer]
                - np.outer(sx, mu)
                - np.outer(mu, sx)
                + self.n * np.outer(mu, mu)
            ) / np.outer(scale[layer], scale[layer])
            _, basis = la.eigh(gram, subset_by_index=(len(mu) - rank, len(mu) - 1))
            basis = basis[:, ::-1].copy()
            signs = np.sign(basis[np.abs(basis).argmax(0), np.arange(rank)])
            signs[signs == 0] = 1
            basis *= signs
            bases.append(basis)
        return SpectralMapping(mean, scale, np.stack(bases))


class AnalyticReadout:
    """One head's two maps, statistics and equal-weight folded coefficients."""

    def __init__(self, mappings, statistics=None, coefficients=None):
        if set(mappings) not in ({"cls", "mean"}, {"global", "mean"}):
            raise ValueError("Supply a CLS or global map alongside the patch-mean map")
        self.branches = ("cls" if "cls" in mappings else "global", "mean")
        self.mappings = mappings
        self.statistics = statistics or {}
        self.coefficients = coefficients

    @classmethod
    def from_mapping_file(cls, path):
        with np.load(path, allow_pickle=False) as data:
            branches = ("global", "mean") if "global_basis" in data else BRANCHES
            mappings = {
                b: SpectralMapping(
                    *(data[f"{b}_{key}"] for key in ("mean", "scale", "basis"))
                )
                for b in branches
            }
        return cls(mappings)

    def update(self, features, labels):
        for branch in self.branches:
            extra = SufficientStatistics.from_rows(
                self.mappings[branch].project(features[branch]), labels
            )
            self.statistics[branch] = (
                self.statistics[branch].add(extra)
                if branch in self.statistics
                else extra
            )

    def solve(self, ridge_lambda=config.RIDGE_LAMBDA):
        coefficients, audits, bias = {}, {}, 0.0
        for branch in self.branches:
            if branch not in self.statistics:
                raise ValueError("Source sufficient statistics required before solving")
            weights, audits[branch] = self.statistics[branch].solve(
                ridge_lambda=ridge_lambda
            )
            beta, offset = self.mappings[branch].fold(weights)
            coefficients[branch] = 0.5 * beta
            bias += 0.5 * offset
        coefficients["bias"] = np.asarray(bias)
        self.coefficients = coefficients
        return audits
