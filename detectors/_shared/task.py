"""An image task composes a model, preprocessing and a method-owned trainer."""

import torch

from data.loader import ImageCollator

from .transforms import ImageTransform


class ImageClassificationTask:
    def __init__(self, model, *, construction, preprocessing, origin, ready=False):
        self.model = model
        self.construction = construction
        self.preprocessing = preprocessing
        self.origin = origin
        self._ready = ready
        self.collate_fn = self.get_collate_fn("evaluation")

    def get_collate_fn(self, stage):
        if stage not in ("train", "validation", "evaluation"):
            raise ValueError(f"Unknown image preprocessing stage: {stage}")
        return ImageCollator(
            transform=ImageTransform(self.preprocessing, stage == "train")
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        from .trainer import ImageClassificationTrainer

        report = ImageClassificationTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import model weights before prediction")
        self.model.to(runtime.device).eval()
        with runtime.autocast():
            logits = self.model(images.to(runtime.device, non_blocking=True))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(
                    "Binary classification task requires two logits per image"
                )
            return logits.softmax(1)[:, 1]

    def state_dict(self):
        if not self._ready:
            raise ValueError("Cannot save an unfitted classification task")
        return {
            "version": 1,
            "construction": self.construction,
            "preprocessing": self.preprocessing,
            "origin": self.origin,
            "state_dict": {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            },
        }

    @staticmethod
    def validate_state(state):
        if state.get("version") != 1:
            raise ValueError("Unsupported image task checkpoint version")
