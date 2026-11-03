"""Source-only grouped selection policy, kept outside the fusion module."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .fusion import LDAFusion


def calibrate_fusion(scores_by_arm, labels, *, groups, folds, score_id):
    """Return a fitted LDAFusion and JSON-safe source selection audit.

    Inputs contain only SOURCE calibration videos; exclude source holdout and
    targets before calling. Each original-video connected group must belong
    to one fold. Search mean/mean_std and eta={0,.25,.5,.75,1}. Average fold
    AUCs by positive-negative pair counts. Ties within 1e-12 prefer larger eta,
    then mean_std, exactly as in the frozen confirmation experiment.
    """
    if not scores_by_arm or set(scores_by_arm) - {"mean", "mean_std"}:
        raise ValueError("provide mean and/or mean_std calibration tensors")
    scores = dict(scores_by_arm)
    size = len(next(iter(scores.values())))
    for x in scores.values():
        LDAFusion._validate_scores(x)
        if len(x) != size:
            raise ValueError("all arms must have the same ordered videos")
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
    traces = []
    for arm in ("mean", "mean_std"):
        if arm not in scores:
            continue
        x = scores[arm]
        for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows = []
            for fold in fold_ids:
                test = folds == fold
                train_ix = torch.as_tensor(np.flatnonzero(~test), device=x.device)
                test_ix = torch.as_tensor(np.flatnonzero(test), device=x.device)
                ly = labels.to(x.device)
                head = LDAFusion({"eta": eta, "arm": arm}).to(device=x.device)
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
            traces.append({"arm": arm, "eta": eta, "cv_auc": value, "folds": rows})
    best = max(t["cv_auc"] for t in traces)
    chosen = max(
        (t for t in traces if best - t["cv_auc"] <= 1e-12),
        key=lambda t: (t["eta"], t["arm"] == "mean_std"),
    )
    x = scores[chosen["arm"]]
    head = LDAFusion({"eta": chosen["eta"], "arm": chosen["arm"]}).to(device=x.device)
    head.fit(x, labels, score_id=score_id)
    return head, {
        "arm": chosen["arm"],
        "eta": chosen["eta"],
        "cv_auc": chosen["cv_auc"],
        "traces": traces,
    }
