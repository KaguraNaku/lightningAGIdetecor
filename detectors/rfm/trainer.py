"""RFM retains the source's unscaled FAM parameter gradients before CE backward."""

from detectors._shared.trainer import ImageClassificationTrainer

from .augmentation import RFMAugmenter


class RFMTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin)
        self.augmentation = RFMAugmenter(model.backbone)

    def compute_loss(self, images, labels):
        augmented = self.augmentation.apply(images)
        # Native standard training zeroes before forward. FAM backward's
        # parameter gradients survive and accumulate with CE. Under native AMP,
        # only CE is scaled; preserving that ordering also preserves this quirk.
        return super().compute_loss(augmented, labels)
