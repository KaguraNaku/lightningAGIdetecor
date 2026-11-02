"""Small adapter contract; fitting algorithms stay with their model implementations.

A configured `module:factory` is called as factory(settings, state=None).
Checkpoint state must use tensors and ordinary Python values compatible with
torch.load(weights_only=True). Larger scores always mean more likely fake.
Collators return pixel_values, labels, metadata and filter failed (None) samples;
an entirely failed batch returns None. See data.loader.ImageCollator for an example.

An optional prepare_dataset(dataset, stage=...) adapts a resolved manifest before
sampling and worker creation. A task may retain an explicit source subset or
attach required supervision while preserving manifest/source identities. Tasks
without this hook keep the original dataset. Selection policy and excluded-source
coverage belong in the method's protocol receipts.

An adapter may additionally implement predict_batch(batch, runtime=...) when
the original algorithm observes labels, metadata or aligned auxiliary inputs
during evaluation. The evaluator prefers this route; image-only adapters retain
predict(images).
"""

from pathlib import Path
from typing import Callable, Protocol

import torch


class ExperimentTask(Protocol):
    def get_collate_fn(self, stage: str) -> Callable:
        """Return a worker-safe collator for train, validation or evaluation.

        Legacy adapters with a single collate_fn remain supported by the engine.
        """
        ...

    def fit(
        self, loader, *, validation_loader=None, runtime, settings: dict, output_dir: Path
    ) -> dict:
        """Fit and return JSON-serializable diagnostics; process every input record."""
        ...

    def predict(self, images: torch.Tensor, *, runtime) -> torch.Tensor:
        """Return one finite raw fake score per image, with shape (N,)."""
        ...

    def state_dict(self) -> dict:
        """Return self-contained model, preprocessing and construction state."""
        ...
