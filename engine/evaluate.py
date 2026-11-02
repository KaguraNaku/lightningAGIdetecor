"""One evaluation loop for every task adapter, with sample-aligned outputs."""

import json
import logging
import os
from pathlib import Path

import torch

from .metrics import MetricAccumulator
from .runtime import atomic_save

logger = logging.getLogger(__name__)


def coverage(dataset, successful_ids, output_dir):
    """Reconcile actual observations independently of log levels and worker state."""
    expected = {record.id for record in dataset.records}
    if not successful_ids <= expected:
        raise ValueError("Observed sample IDs outside the input manifest")
    skipped = [
        record.id for record in dataset.records if record.id not in successful_ids
    ]
    path = Path(output_dir) / "skipped-samples.jsonl"
    with path.open("w") as stream:
        for sample_id in skipped:
            stream.write(json.dumps({"id": sample_id}) + "\n")
    return {
        "expected": len(expected),
        "successful": len(successful_ids),
        "skipped": len(skipped),
    }


@torch.inference_mode()
def evaluate(task, loader, *, runtime, settings, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulator = MetricAccumulator(
        threshold=settings["threshold"],
        video_aggregation=settings["video_aggregation"],
        group_by=settings["group_by"],
        prediction_unit=getattr(task, "prediction_unit", "image"),
    )
    seen = set()
    expected_records = {record.id: record for record in loader.dataset.records}
    temporary = output_dir / "predictions.partial.jsonl"
    with temporary.open("w") as stream:
        for batch in loader:
            predict_batch = getattr(task, "predict_batch", None)
            scores = (
                predict_batch(batch, runtime=runtime)
                if callable(predict_batch)
                else task.predict(batch["pixel_values"], runtime=runtime)
            )
            count = len(batch["metadata"])
            labels = batch["labels"].detach().cpu()
            if not isinstance(scores, torch.Tensor) or scores.shape != (count,):
                raise ValueError(
                    "Task predict must return a score tensor with shape (N,)"
                )
            if labels.shape != (count,) or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Batch labels must be aligned binary labels")
            scores = scores.detach().double().cpu()
            if not torch.isfinite(scores).all():
                raise ValueError("Task returned nonfinite scores")
            for metadata, label, score in zip(
                batch["metadata"], labels.tolist(), scores.tolist(), strict=True
            ):
                sample_id = metadata["id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate evaluated sample: {sample_id}")
                source = expected_records.get(sample_id)
                if (
                    source is None
                    or source.label != label
                    or source.dataset != metadata["dataset"]
                    or source.split != metadata["split"]
                ):
                    raise ValueError(
                        f"Prediction identity/label differs from manifest: {sample_id}"
                    )
                seen.add(sample_id)
                record = {**metadata, "label": label, "score": score}
                if accumulator.prediction_unit != "image":
                    record.setdefault("prediction_unit", accumulator.prediction_unit)
                accumulator.add(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    report = {
        "coverage": coverage(loader.dataset, seen, output_dir),
        **accumulator.result(),
    }
    os.replace(temporary, output_dir / "predictions.jsonl")
    atomic_save(output_dir / "metrics.json", report)
    logger.info(
        "Evaluated %d samples; skipped %d", len(seen), report["coverage"]["skipped"]
    )
    return report
