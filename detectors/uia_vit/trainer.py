"""Native CE/PCL cadence, labeled evaluation and coupled statistics snapshots."""

from torch import nn
from torch.nn import functional as F

from detectors._shared.trainer import ImageClassificationTrainer

from .loss import PatchConsistencyLoss


class UIAObjective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pcl = PatchConsistencyLoss(model)

    def forward(self, outputs, labels, *, detector):
        ce = F.cross_entropy(outputs["cls"], labels)
        if (
            detector.batch_cnt <= detector.batch_per_epoch
            or not detector.model.training
        ):
            return ce
        statistics = detector.statistics
        if any(
            getattr(statistics, name) is None
            for name in (
                "real_feature_mean",
                "real_inv_covariance",
                "fake_feature_mean",
                "fake_inv_covariance",
            )
        ):
            raise ValueError("UIA PCL requires previously fitted real/fake statistics")
        pcl = self.pcl(
            outputs["attention_map_real"],
            outputs["attention_map_fake"],
            outputs["feat"],
            statistics.real_feature_mean,
            statistics.real_inv_covariance,
            statistics.fake_feature_mean,
            statistics.fake_inv_covariance,
            labels,
        )
        core = detector.model
        return (
            ce
            + 0.06 * pcl
            + 0.05 * (1 / core.c_real.abs() + 1 / core.c_fake.abs())
            + 0.5 * core.c_cross.abs()
        )


class LabeledScorer:
    def __init__(self, model):
        self.model = model

    def predict_batch(self, batch, *, runtime):
        with runtime.autocast():
            return self.model.forward_outputs(
                batch["pixel_values"].to(runtime.device, non_blocking=True),
                batch["labels"].to(runtime.device, non_blocking=True),
            )["cls"].softmax(1)[:, 1]


class UIATrainer(ImageClassificationTrainer):
    def __init__(self, model, objective, *, origin=None):
        super().__init__(model, origin=origin, objective=objective)

    def build_optimizer(self, settings):
        return super().build_optimizer(
            {"learning_rate": 3e-5, "weight_decay": 1e-5, **settings}
        )

    def compute_loss(self, images, labels):
        outputs = self.model.forward_outputs(images, labels)
        return self.objective(outputs, labels, detector=self.model)

    def make_scorer(self):
        return LabeledScorer(self.model)

    def method_state_dict(self):
        return self.model.method_state_dict()

    def load_method_state_dict(self, state):
        self.model.load_method_state_dict(state)
