"""Source-only audit of fixed Global + (Mean+Std) LDA, without search."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .fusion import LDAFusion


def calibrate_fusion(scores, labels, *, groups, folds, score_id, eps=1e-6):
    """Fit Global/Mean+Std scores and return a source validation audit.

    Each row is one SOURCE calibration video, columns [Global, Mean+Std].
    Exclude source holdout and targets before calling. Each original-video
    connected group belongs to one fold. Cross-validation reports the fixed
    method's pair-count-weighted held-out AUC; it selects no hyperparameters.
    The returned module is fitted on all supplied calibration videos.
    """
    LDAFusion._validate_scores(scores)
    size = len(scores)
    if not isinstance(labels, torch.Tensor) or labels.shape != (size,):
        raise ValueError("labels must contain one value per calibration video")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must contain only real=0 and fake=1")
    y = labels.detach().cpu().numpy()
    groups, folds = np.asarray(groups), np.asarray(folds)
    if groups.shape != (size,) or folds.shape != (size,):
        raise ValueError("groups and folds must have shape (N,)")
    if groups.dtype.kind in "fc" and not np.isfinite(groups).all():
        raise ValueError("groups must not contain missing identities")
    if groups.dtype.kind not in "iufUS":
        raise ValueError("groups must contain finite numbers or identity strings")
    if groups.dtype.kind in "US" and (groups == "").any():
        raise ValueError("groups must not contain empty identities")
    if folds.dtype.kind not in "iu" or (folds < 0).any() or len(np.unique(folds)) < 2:
        raise ValueError("need at least two nonnegative integer folds")
    fold_ids = np.unique(folds)
    for fold in fold_ids:
        test = folds == fold
        if np.intersect1d(groups[test], groups[~test]).size:
            raise ValueError("an original-video group crosses validation folds")
        if len(np.unique(y[test])) != 2:
            raise ValueError("every validation fold must contain both classes")
    rows = []
    x, ly = scores, labels.to(scores.device)
    for fold in fold_ids:
        test = folds == fold
        train_ix = torch.as_tensor(np.flatnonzero(~test), device=x.device)
        test_ix = torch.as_tensor(np.flatnonzero(test), device=x.device)
        head = LDAFusion({"eps": eps}).to(device=x.device)
        head.fit(x[train_ix], ly[train_ix], score_id=score_id)
        prediction = head(x[test_ix]).detach().cpu().numpy()
        n1 = int(y[test].sum())
        rows.append(
            {
                "fold": int(fold),
                "train_indices": np.flatnonzero(~test).tolist(),
                "test_indices": np.flatnonzero(test).tolist(),
                "auc": float(roc_auc_score(y[test], prediction)),
                "pairs": n1 * (int(test.sum()) - n1),
                "weights": head.weights.cpu().tolist(),
                "bias": float(head.bias),
            }
        )
    value = float(
        np.average([r["auc"] for r in rows], weights=[r["pairs"] for r in rows])
    )
    head = LDAFusion({"eps": eps}).to(device=x.device)
    head.fit(x, labels, score_id=score_id)
    return head, {
        "arm": "mean_std",
        "inputs": ["global", "mean_std"],
        "epsilon": float(eps),
        "selection_performed": False,
        "cv_auc": value,
        "folds": rows,
    }
