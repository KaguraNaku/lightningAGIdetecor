"""Whole-batch clip CE with native augmented source validation and Adam."""

import torch

from detectors._shared.trainer import ImageClassificationTrainer
from detectors.i3d.trainer import I3DTrainer
from .dataset import VideoMAEDataset


class VideoMAETrainer(I3DTrainer):
    def __init__(self, model, *, origin=None):
        ImageClassificationTrainer.__init__(self, model, origin=origin, objective=torch.nn.CrossEntropyLoss())

    def make_scorer(self):
        # Native output is two logits, not I3D's single sigmoid probability.
        return ImageClassificationTrainer.make_scorer(self)

    @staticmethod
    def validate_inputs(loader, validation_loader, monitor, *, training_labels=None):
        if validation_loader is None:
            raise ValueError("VideoMAE requires source-domain fit.validation_input")
        if not isinstance(loader.dataset, VideoMAEDataset) or not isinstance(validation_loader.dataset, VideoMAEDataset):
            raise ValueError("VideoMAE needs its landmark-aware complete-frame clip adapters")
        I3DTrainer.validate_inputs(loader, validation_loader, monitor, training_labels=training_labels)

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        # Keep native inclusive0..10 epochs,whole batch8/final partial batch,
        # all trainable weights and guarded retained saved tensors. Numeric
        # full-encoder backward/CLI parity is still a separate validation gate.
        return super().fit(loader, validation_loader=validation_loader, runtime=runtime,
                           settings={"epochs": 11, **settings}, output_dir=output_dir)
