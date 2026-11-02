"""Flatten declared primary/video metrics without substituting missing units."""

from .units import validate_prediction_unit


def summary_rows(reports):
    rows = []
    for name, report in reports.items():
        primary = validate_prediction_unit(report.get("prediction_unit", "image"))
        metrics = report.get(primary)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"Summary requires the declared {primary} metrics")
        other = "clip" if primary == "image" else "image"
        if report.get(other) is not None:
            raise ValueError("Summary cannot relabel mixed primary prediction units")
        video = report.get("video")
        if video is not None and not isinstance(video, dict):
            raise ValueError("Video metrics must be a dictionary or None")
        row = {"input": name, **report["coverage"]}
        metric_names = dict.fromkeys((*metrics, *(video or {})))
        for unit, values in ((primary, metrics), ("video", video or {})):
            row.update({f"{unit}_{key}": values.get(key) for key in metric_names})
        rows.append(row)
    return rows
