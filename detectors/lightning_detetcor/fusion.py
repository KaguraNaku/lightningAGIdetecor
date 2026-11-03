"""Closed-form two-score LDA fusion, calibrated on source videos only.

Inputs are frozen global/spatial branch scores, not probabilities. For video
calibration, average each branch over frames first. Apply the frozen affine
head per frame or per video; these operations commute. No target-label fitting,
ReLU, sigmoid, PCA search, or backbone changes are performed here.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


ETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = 1e-6
ARMS = ("mean", "mean_std")


def _scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite (N, 2) array")
    return scores


def _labels(labels, size):
    labels = np.asarray(labels)
    if labels.shape != (size,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must have shape (N,), with real=0 and fake=1")
    return labels.astype(np.int64)


def fit_lda(scores, labels, eta=0.0):
    """Fit a JSON-serializable head on source calibration video scores.

    Real-class standardization uses ddof=0 and std+1e-6. Class covariance
    matrices use ddof=0 and equal class weight, regardless of class counts.
    eta shrinks the pooled covariance toward its diagonal; eta=0 is full
    covariance LDA. A fixed 1e-6 diagonal stabilizer is always retained.
    The Fisher direction is L1-normalized and folded into raw-score weights.
    Negative weights are allowed; higher output denotes fake by source fit.
    """
    x = _scores(scores)
    y = _labels(labels, len(x))
    if min(np.count_nonzero(y == label) for label in (0, 1)) < 2:
        raise ValueError("need at least two real and two fake calibration videos")
    if eta not in ETAS:
        raise ValueError(f"eta must be one of {ETAS}")
    mean = x[y == 0].mean(axis=0)
    scale = x[y == 0].std(axis=0, ddof=0) + EPS
    z = (x - mean) / scale
    real, fake = z[y == 0], z[y == 1]
    delta = fake.mean(axis=0) - real.mean(axis=0)
    covariance = (
        np.cov(real, rowvar=False, bias=True) + np.cov(fake, rowvar=False, bias=True)
    ) / 2
    shrunk = (
        (1 - eta) * covariance + eta * np.diag(np.diag(covariance)) + EPS * np.eye(2)
    )
    weights = np.linalg.solve(shrunk, delta)
    norm = np.abs(weights).sum()
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("no finite, nonzero Fisher direction")
    weights /= norm
    raw = weights / scale
    bias = float(-mean @ raw)
    if not np.isfinite(raw).all() or not np.isfinite(bias):
        raise ValueError("fitted affine head must be finite")
    return {
        "eta": float(eta),
        "epsilon": EPS,
        "real_mean": mean.tolist(),
        "real_scale": scale.tolist(),
        "delta": delta.tolist(),
        "within_covariance": covariance.tolist(),
        "shrunk_covariance": shrunk.tolist(),
        "normalized_weights": weights.tolist(),
        "raw_weights": raw.tolist(),
        "raw_bias": bias,
        "angle_raw_degrees": float(np.rad2deg(np.arctan2(raw[1], raw[0]))),
        "real_count": int((y == 0).sum()),
        "fake_count": int((y == 1).sum()),
    }


def fuse_lda(scores, head):
    """Apply a frozen head; no labels, refitting, or score-direction flipping."""
    x = _scores(scores)
    weights = np.asarray(head["raw_weights"], dtype=np.float64)
    bias = np.asarray(head["raw_bias"], dtype=np.float64)
    if (
        weights.shape != (2,)
        or bias.shape != ()
        or not np.isfinite(weights).all()
        or not np.isfinite(bias)
    ):
        raise ValueError(
            "head must contain two finite weights and a finite scalar bias"
        )
    with np.errstate(over="raise", invalid="raise"):
        return x @ weights + bias


def select_lda(scores_by_arm, labels, *, groups, folds):
    """Select arm/eta using grouped source-only video cross-validation.

    Supply calibration videos only, with external holdout/targets excluded.
    Each connected original-video group must belong to one validation fold;
    callers must build groups including all related manipulation variants.
    folds are explicit integers (at least two distinct values). Each candidate
    fits normalization/covariance on its training folds. Selection maximizes
    pair-count-weighted mean fold AUC, not pooled cross-fold scores. Ties within
    1e-12 prefer larger eta, then mean_std, matching the frozen experiment.
    The final head is fitted on all supplied calibration videos. Save the
    returned dictionary as JSON before evaluating held-out or target samples.
    """
    if not scores_by_arm or set(scores_by_arm) - set(ARMS):
        raise ValueError("provide mean and/or mean_std score arrays")
    scores = {arm: _scores(x) for arm, x in scores_by_arm.items()}
    size = len(next(iter(scores.values())))
    if any(len(x) != size for x in scores.values()):
        raise ValueError("all arms must describe the same ordered videos")
    y = _labels(labels, size)
    groups, folds = np.asarray(groups), np.asarray(folds)
    if groups.shape != (size,) or folds.shape != (size,):
        raise ValueError("groups and folds must have shape (N,)")
    if folds.dtype.kind not in "iu" or (folds < 0).any():
        raise ValueError("folds must contain nonnegative integers")
    fold_ids = np.unique(folds)
    if len(fold_ids) < 2:
        raise ValueError("need at least two validation folds")
    for fold in fold_ids:
        test = folds == fold
        if np.intersect1d(groups[test], groups[~test]).size:
            raise ValueError("a source group crosses validation folds")
        if len(np.unique(y[test])) != 2:
            raise ValueError("every validation fold must contain both classes")
    traces = []
    for arm in ARMS:
        if arm not in scores:
            continue
        for eta in ETAS:
            rows = []
            for fold in fold_ids:
                test = folds == fold
                head = fit_lda(scores[arm][~test], y[~test], eta)
                n1 = int(y[test].sum())
                rows.append(
                    {
                        "fold": int(fold),
                        "train_indices": np.flatnonzero(~test).tolist(),
                        "test_indices": np.flatnonzero(test).tolist(),
                        "head": head,
                        "auc": float(
                            roc_auc_score(y[test], fuse_lda(scores[arm][test], head))
                        ),
                        "pairs": n1 * (int(test.sum()) - n1),
                    }
                )
            value = float(
                np.average(
                    [row["auc"] for row in rows], weights=[row["pairs"] for row in rows]
                )
            )
            traces.append({"arm": arm, "eta": eta, "cv_auc": value, "folds": rows})
    best = max(row["cv_auc"] for row in traces)
    chosen = max(
        (row for row in traces if best - row["cv_auc"] <= 1e-12),
        key=lambda row: (row["eta"], row["arm"] == "mean_std"),
    )
    return {
        "arm": chosen["arm"],
        "eta": chosen["eta"],
        "cv_auc": chosen["cv_auc"],
        "head": fit_lda(scores[chosen["arm"]], y, chosen["eta"]),
        "traces": traces,
    }
