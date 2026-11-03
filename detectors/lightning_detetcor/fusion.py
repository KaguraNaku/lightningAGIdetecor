"""Frozen, source-calibrated two-score Fisher fusion as a registered module."""

import math

import torch
from torch import nn


class LDAFusion(nn.Module):
    """Affine score head with fitted statistics in ``state_dict`` buffers.

    Columns are Global and Mean+Std branch scores; real=0, fake=1. Calibration uses
    video means, ddof=0 class covariances with equal class weight, and source
    real-class std+epsilon. Only epsilon*I is added for numerical stability;
    no covariance shrinkage or arm selection is performed. Negative Fisher
    coefficients are allowed. Source validation lives outside inference.
    """

    VERSION = 2

    def __init__(self, config):
        super().__init__()
        if set(config) - {"eps", "arm"}:
            raise ValueError("unsupported fusion configuration; shrinkage was removed")
        self.epsilon = float(config.get("eps", 1e-6))
        self.arm = "mean_std"
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("eps must be finite and positive")
        if config.get("arm", "mean_std") != "mean_std":
            raise ValueError("fusion is fixed to Global + (Mean+Std)")
        self.score_id = None
        for name in ("weights", "normalized_weights", "real_mean", "delta"):
            self.register_buffer(name, torch.zeros(2, dtype=torch.float64))
        self.register_buffer("real_scale", torch.ones(2, dtype=torch.float64))
        for name in ("within_covariance", "regularized_covariance"):
            self.register_buffer(name, torch.zeros(2, 2, dtype=torch.float64))
        self.register_buffer("bias", torch.zeros((), dtype=torch.float64))
        self.register_buffer("class_counts", torch.zeros(2, dtype=torch.int64))

    @staticmethod
    def _validate_score_id(score_id, *, allow_none=False):
        if allow_none and score_id is None:
            return
        if not isinstance(score_id, str) or not score_id.strip():
            raise ValueError("score_id must be a nonempty pipeline fingerprint")

    def get_extra_state(self):
        return {
            "version": self.VERSION,
            "epsilon": self.epsilon,
            "arm": self.arm,
            "score_id": self.score_id,
        }

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError(
                "saved fusion metadata is incompatible; refit Global + (Mean+Std)"
            )
        for name in ("version", "epsilon", "arm"):
            if (
                type(state[name]) is not type(expected[name])
                or state[name] != expected[name]
            ):
                raise ValueError(f"saved fusion {name} does not match configuration")
        empty = bool((self.class_counts == 0).all())
        if not empty and not bool((self.class_counts >= 2).all()):
            raise ValueError("saved fusion class counts are inconsistent")
        self._validate_score_id(state["score_id"], allow_none=empty)
        if any(not bool(torch.isfinite(b).all()) for b in self.buffers()):
            raise ValueError("saved fusion buffers must be finite")
        if not bool((self.real_scale > 0).all()):
            raise ValueError("saved fusion scales must be positive")
        self.score_id = state["score_id"]

    @staticmethod
    def _validate_scores(scores):
        if (
            not isinstance(scores, torch.Tensor)
            or scores.ndim != 2
            or scores.shape[1] != 2
        ):
            raise ValueError("scores must be a finite (N, 2) tensor")
        if scores.is_complex() or not bool(torch.isfinite(scores).all()):
            raise ValueError("scores must be finite real values")

    @torch.no_grad()
    def fit(self, scores, labels, *, score_id):
        """Fit source video scores atomically; existing state survives rejection.

        score_id must identify the frozen upstream models, readouts, mappings,
        heads, preprocessing and branch order. Refitting upstream branches
        requires recalibrating this fusion head as well.
        """
        self._validate_score_id(score_id)
        self._validate_scores(scores)
        if not isinstance(labels, torch.Tensor) or labels.shape != (len(scores),):
            raise ValueError("labels must contain one raw 0/1 value per video")
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("labels must contain only real=0 and fake=1")
        x = scores.to(device=self.weights.device, dtype=torch.float64)
        y = labels.to(device=x.device)
        counts = torch.stack([(y == label).sum() for label in (0, 1)])
        if not bool((counts >= 2).all()):
            raise ValueError("need at least two real and two fake calibration videos")
        mean = x[y == 0].mean(0)
        scale = x[y == 0].std(0, correction=0) + self.epsilon
        standardized = (x - mean) / scale
        real, fake = standardized[y == 0], standardized[y == 1]
        delta = fake.mean(0) - real.mean(0)
        r, f = real - real.mean(0), fake - fake.mean(0)
        covariance = (r.T @ r / len(r) + f.T @ f / len(f)) / 2
        regularized = covariance + self.epsilon * torch.eye(
            2, dtype=x.dtype, device=x.device
        )
        w = torch.linalg.solve(regularized, delta)
        norm = w.abs().sum()
        if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
            raise ValueError("no finite, nonzero Fisher direction")
        w = w / norm
        raw = w / scale
        candidate = {
            "weights": raw,
            "bias": -mean @ raw,
            "normalized_weights": w,
            "real_mean": mean,
            "real_scale": scale,
            "delta": delta,
            "within_covariance": covariance,
            "regularized_covariance": regularized,
            "class_counts": counts,
        }
        converted = {
            name: value.to(getattr(self, name)) for name, value in candidate.items()
        }
        if any(not bool(torch.isfinite(value).all()) for value in converted.values()):
            raise ValueError("fitted fusion state must be finite")
        if not bool((converted["real_scale"] > 0).all()):
            raise ValueError("fitted fusion scales must be positive")
        for name, value in converted.items():
            getattr(self, name).copy_(value)
        self.score_id = score_id
        return self

    def forward(self, scores, *, score_id=None):
        """Apply the frozen affine head. No fitting, ReLU or sign reversal."""
        if not bool((self.class_counts >= 2).all()):
            raise ValueError("fit before prediction")
        if score_id is not None:
            self._validate_score_id(score_id)
            if score_id != self.score_id:
                raise ValueError("upstream score pipeline changed; recalibrate fusion")
        self._validate_scores(scores)
        return scores.to(self.weights) @ self.weights + self.bias
