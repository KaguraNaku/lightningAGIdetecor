"""Spectral mapping, Ridge fitting, and equal-weight global and patch-mean scores."""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import scipy.linalg as la

BRANCHES = ("cls", "mean")
PROJECTION_RANK = 128
DEFAULT_RIDGE_LAMBDA = .15


def positive_ridge_lambda(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    return float(value)


@dataclass
class SufficientStatistics:
    g: np.ndarray
    h: np.ndarray
    sz: np.ndarray
    sy: float
    n: int

    @classmethod
    def from_rows(cls, z, y):
        z, y = np.asarray(z, dtype=np.float64), np.asarray(y)
        if (z.ndim != 2 or y.shape != (len(z),)
                or (y.dtype != np.bool_ and not np.isin(y, [0, 1]).all())):
            raise ValueError("Aligned binary-labelled projected rows required")
        y = y.astype(np.float64, copy=False)
        return cls(z.T @ z, z.T @ y, z.sum(0), float(y.sum()), len(y))

    @classmethod
    def from_state(cls, state):
        return cls(*(np.asarray(state[k], dtype=np.float64) for k in ("g", "h", "sz")),
                   float(state["sy"]), int(state["n"]))

    def state_dict(self):
        return {"g": self.g, "h": self.h, "sz": self.sz,
                "sy": np.asarray(self.sy), "n": np.asarray(self.n, dtype=np.int64)}

    def add(self, other):
        if self.g.shape != other.g.shape:
            raise ValueError("Cannot combine statistics under different feature dimensions")
        return type(self)(self.g + other.g, self.h + other.h, self.sz + other.sz,
                          self.sy + other.sy, self.n + other.n)

    def solve(self, *, ridge_lambda=DEFAULT_RIDGE_LAMBDA):
        if not 0 < self.sy < self.n:
            raise ValueError("Ridge fitting requires both classes")
        ridge_lambda = positive_ridge_lambda(ridge_lambda)
        q = self.h - self.sy / self.n * self.sz
        # Minimize ||Z w - (y - mean(y))||² / n + ridge_lambda * ||w||².
        penalty = self.n * ridge_lambda
        system = self.g.copy()
        system.flat[::len(system) + 1] += penalty
        if not np.isfinite(q).all():
            raise ValueError("Ridge right-hand side must be finite")
        factor = la.cho_factor(system, lower=True)
        w = la.cho_solve(factor, q, check_finite=False)
        residual = float(la.norm(self.g @ w + penalty * w - q) / max(la.norm(q), 1e-30))
        if not math.isfinite(residual) or residual >= 1e-8 or not np.isfinite(w).all():
            raise ValueError("Analytic ridge normal-equation check failed")
        return w, {"relative_normal_equation_residual": residual, "n": self.n,
                   "n_fake": int(self.sy), "label_mean": self.sy / self.n,
                   "ridge_lambda": ridge_lambda,
                   "system_diagonal_penalty": penalty}


@dataclass
class SpectralMapping:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray

    def __post_init__(self):
        self.mean, self.scale, self.basis = [np.asarray(x, dtype=np.float64)
                                            for x in (self.mean, self.scale, self.basis)]
        if (self.mean.ndim != 2 or self.scale.shape != self.mean.shape
                or self.basis.ndim != 3 or self.basis.shape[:2] != self.mean.shape
                or not all(np.isfinite(x).all() for x in (self.mean, self.scale, self.basis))
                or not (self.scale > 0).all()):
            raise ValueError("Invalid frozen spectral mapping")

    @property
    def rank(self):
        return self.basis.shape[-1]

    def project(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != self.mean.shape:
            raise ValueError("Raw readouts differ from declared mapping")
        return np.einsum('nld,ldr->nlr', (x - self.mean) / self.scale,
                         self.basis, optimize=True).reshape(len(x), -1)

    def fold(self, weights):
        weights = np.asarray(weights).reshape(self.mean.shape[0], self.rank)
        beta = np.einsum('ldr,lr->ld', self.basis, weights) / self.scale
        return beta, -float(np.sum(beta * self.mean))

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale, "basis": self.basis}


class MappingAccumulator:
    """Raw moments for a streaming, two-pass source mapping fit."""
    def __init__(self):
        self.n = self.n_real = 0
        self.sum = self.real_sum = self.real_square = self.gram = None

    def update(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y)
        if (x.ndim != 3 or y.shape != (len(x),)
                or (y.dtype != np.bool_ and not np.isin(y, [0, 1]).all())):
            raise ValueError("Aligned binary readouts required")
        if self.sum is None:
            self.sum = np.zeros(x.shape[1:]); self.real_sum = np.zeros_like(self.sum)
            self.real_square = np.zeros_like(self.sum)
            self.gram = np.zeros((x.shape[1], x.shape[2], x.shape[2]))
        if x.shape[1:] != self.sum.shape:
            raise ValueError("Readout shape changed during mapping fit")
        real = x[y == 0]
        self.n += len(x); self.n_real += len(real)
        self.sum += x.sum(0); self.real_sum += real.sum(0)
        self.real_square += np.square(real).sum(0)
        self.gram += np.einsum('nld,nle->lde', x, x, optimize=True)

    def fit_mapping(self, rank=PROJECTION_RANK):
        if not self.n_real or self.n < rank or not 0 < rank <= self.sum.shape[-1]:
            raise ValueError("Insufficient source rows or genuine references")
        mean = self.real_sum / self.n_real
        scale = np.sqrt(np.maximum(self.real_square / self.n_real - mean**2, 0) + 1e-6)
        bases = []
        for layer in range(len(mean)):
            mu, sx = mean[layer], self.sum[layer]
            gram = (self.gram[layer] - np.outer(sx, mu) - np.outer(mu, sx)
                    + self.n * np.outer(mu, mu)) / np.outer(scale[layer], scale[layer])
            _, basis = la.eigh(gram, subset_by_index=(len(mu) - rank, len(mu) - 1))
            basis = basis[:, ::-1].copy()
            signs = np.sign(basis[np.abs(basis).argmax(0), np.arange(rank)])
            signs[signs == 0] = 1; basis *= signs
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
                b: SpectralMapping(*(data[f"{b}_{key}"] for key in ("mean", "scale", "basis")))
                for b in branches
            }
        return cls(mappings)

    def update(self, features, labels):
        for branch in self.branches:
            extra = SufficientStatistics.from_rows(self.mappings[branch].project(features[branch]), labels)
            self.statistics[branch] = (self.statistics[branch].add(extra)
                                      if branch in self.statistics else extra)

    def solve(self, ridge_lambda=DEFAULT_RIDGE_LAMBDA):
        coefficients, audits, bias = {}, {}, 0.
        for branch in self.branches:
            if branch not in self.statistics:
                raise ValueError("Source sufficient statistics required before solving")
            weights, audits[branch] = self.statistics[branch].solve(ridge_lambda=ridge_lambda)
            beta, offset = self.mappings[branch].fold(weights)
            coefficients[branch] = .5 * beta
            bias += .5 * offset
        coefficients['bias'] = np.asarray(bias)
        self.coefficients = coefficients
        return audits
