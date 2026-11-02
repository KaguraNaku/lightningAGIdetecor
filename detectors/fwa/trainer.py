"""Whole effective2B FWA training with its original paired protocol."""

import importlib.metadata

from detectors._shared.paired import PairedImageTrainer
from engine.runtime import atomic_save
from .generation import FWAPairCollator


class FWATrainer(PairedImageTrainer):
    fallback_statistic = "no_face_fallbacks"

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.collate_fn, FWAPairCollator):
            raise ValueError("FWA training requires the native generated-pair collator")
        from pathlib import Path

        atomic_save(
            Path(output_dir) / "paired-protocol.json",
            {
                "source_batch_size": loader.batch_size,
                "effective_model_batch_size": loader.batch_size * 2,
                "ordering": "generated_all_then_original_all_then_native_python_shuffle",
                "loss_denominator": "actual generated+original model images",
                "coverage_unit": "source manifest images",
                "generation": "unconditional native augmentations, including discarded augmentation RNG draws",
                "predictor": loader.collate_fn.predictor,
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ("dlib", "scikit-image", "albumentations", "scipy")
                },
                "exact_training_resume_supported": False,
            },
        )
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 11,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output_dir,
        )
