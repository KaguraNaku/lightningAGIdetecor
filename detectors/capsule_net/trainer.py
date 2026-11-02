"""Native capsule CE sums four probability-input terms, retaining gradients."""

from torch import nn

from detectors._shared.trainer import ImageClassificationTrainer


class CapsuleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        loss = self.cross_entropy_loss(inputs[:, 0, :], labels)
        for index in range(inputs.size(1) - 1):
            loss = loss + self.cross_entropy_loss(inputs[:, index + 1, :], labels)
        return loss


class CapsuleTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=CapsuleLoss())

    def compute_loss(self, images, labels):
        return self.objective(self.model.forward_outputs(images)["classes"], labels)

    def build_optimizer(self, settings):
        return super().build_optimizer(
            {"learning_rate": 8e-5, "weight_decay": 5e-4, **settings}
        )
