"""CORE owns its feature-dependent batch loss; shared fitting retains receipts."""

from detectors._shared.trainer import ImageClassificationTrainer

from .loss import ConsistencyCosineLoss


class CORETrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=ConsistencyCosineLoss())

    def compute_loss(self, images, labels):
        outputs = self.model.forward_outputs(images)
        if outputs["cls"].shape != (len(labels), 2):
            raise ValueError("CORE classification requires (N, 2) logits")
        return self.objective(outputs["core_feat"], outputs["cls"], labels)
