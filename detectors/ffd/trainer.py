"""Configured FFD CE training; no dormant mask objective or scheduler."""

from detectors._shared.trainer import ImageClassificationTrainer


class FFDTrainer(ImageClassificationTrainer):
    def compute_loss(self, images, labels):
        return self.model.loss_terms(self.model.forward_outputs(images), labels)[
            "overall"
        ]

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
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
