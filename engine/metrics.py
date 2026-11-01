"""Binary fake-score metrics and explicit video/metadata aggregation."""

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Metrics require nonempty aligned labels and scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Metrics require binary labels and finite scores")
    predicted = scores >= threshold
    real, fake = labels == 0, labels == 1
    result = {
        "count": len(labels),
        "real": int(real.sum()),
        "fake": int(fake.sum()),
        "threshold": float(threshold),
        "acc": float((predicted == labels).mean()),
        "acc_real": float((~predicted[real]).mean()) if real.any() else None,
        "acc_fake": float(predicted[fake].mean()) if fake.any() else None,
        "auc": None,
        "ap": None,
        "eer": None,
    }
    if real.any() and fake.any():
        result["auc"] = float(roc_auc_score(labels, scores))
        result["ap"] = float(average_precision_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        difference = fpr - fnr
        right = int(np.flatnonzero(difference >= 0)[0])
        left = max(0, right - 1)
        fraction = (
            0.0
            if difference[right] == difference[left]
            else -difference[left] / (difference[right] - difference[left])
        )
        result["eer"] = float(fpr[left] + fraction * (fpr[right] - fpr[left]))
    else:
        result["reason"] = "Ranking metrics require both real and fake samples"
    return result


class MetricAccumulator:
    """Keep scores on CPU for ranking metrics, not image tensors or GPU outputs."""

    def __init__(self, *, threshold=0.0, video_aggregation="mean", group_by=()):
        self.threshold = threshold
        self.video_aggregation = video_aggregation
        self.group_by = group_by
        self.labels, self.scores = [], []
        self.videos, self.groups = {}, {key: {} for key in group_by}
        self.missing_video = 0

    def add(self, record):
        label, score = record["label"], record["score"]
        self.labels.append(label)
        self.scores.append(score)
        if self.video_aggregation == "mean":
            if record.get("video_id"):
                key = (record["dataset"], record["video_id"])
                previous = self.videos.setdefault(key, [label, 0.0, 0])
                if previous[0] != label:
                    raise ValueError(f"Conflicting labels for video {key}")
                previous[1] += score
                previous[2] += 1
            else:
                self.missing_video += 1
        for field in self.group_by:
            value = record
            for part in field.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value is None or value == "":
                continue
            name = (
                value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            )
            labels, scores = self.groups[field].setdefault(name, ([], []))
            labels.append(label)
            scores.append(score)

    def result(self):
        result = {"image": binary_metrics(self.labels, self.scores, self.threshold)}
        result["video"] = None
        if self.videos:
            values = list(self.videos.values())
            result["video"] = binary_metrics(
                [v[0] for v in values], [v[1] / v[2] for v in values], self.threshold
            )
        result["samples_without_video_id"] = self.missing_video
        result["groups"] = {
            field: {
                name: binary_metrics(labels, scores, self.threshold)
                for name, (labels, scores) in groups.items()
            }
            for field, groups in self.groups.items()
        }
        return result
