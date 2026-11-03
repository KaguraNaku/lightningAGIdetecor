"""TALL's native CE and Adam; complete cutout data fitting remains a next gate."""

from detectors._shared.trainer import ImageClassificationTrainer


class TALLTrainer(ImageClassificationTrainer):
    prediction_unit = "clip"

    def build_optimizer(self, settings):
        if settings.get("optimizer", "adam") != "adam":
            raise ValueError("The selected native TALL profile uses Adam")
        if any(not p.requires_grad for p in self.model.parameters()):
            raise ValueError("Native TALL trains every model parameter")
        return super().build_optimizer({"learning_rate": 2e-5, **settings})

    def compute_loss(self, clips, labels):
        outputs = self.model.forward_outputs(clips)
        return self.model.loss_terms(outputs, labels)["overall"]

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        raise NotImplementedError(
            "TALL fitting requires its explicitly selected T4 source-data recipe; "
            "only the independently verified model objective is available yet"
        )
