"""Native auxiliary loss, shared whole-batch fitting and source-only selection."""

from detectors._shared.trainer import ImageClassificationTrainer
from .loss import ForAdaLoss


class ForAdaTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=ForAdaLoss())

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        fields = ("xray", "patch_label", "clip_patch_label", "if_boundary")
        if any(key not in batch for key in fields):
            raise ValueError(
                "ForAda requires its training collator and explicit mask bindings"
            )
        if not (labels == 1).any():
            raise ValueError(
                "Native ForAda contrastive objective is undefined for real-only batches"
            )
        data = {key: batch[key].to(runtime.device) for key in fields}
        data.update(image=images, label=labels)
        return self.objective(self.model.forward_outputs(data, inference=False), data)[
            "overall"
        ]
