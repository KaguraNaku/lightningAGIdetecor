"""An image task composes a model, preprocessing and a method-owned trainer."""

import torch

from data.loader import ImageCollator

from .trainer import ImageClassificationTrainer
from .transforms import ImageTransform


class ImageClassificationTask:
    trainer_class = ImageClassificationTrainer
    default_epochs = None
    default_allow_tf32 = False

    def __init__(self, model, *, preprocessing, origin, construction=None, ready=False):
        self.model = model
        self.construction = construction
        self.preprocessing = preprocessing
        self.origin = origin
        self._ready = ready
        self.collate_fn = self.get_collate_fn("evaluation")

    def get_collate_fn(self, stage):
        if stage not in ("train", "validation", "evaluation"):
            raise ValueError(f"Unknown image preprocessing stage: {stage}")
        return self.build_collator(training=stage == "train")

    def build_collator(self, *, training):
        return ImageCollator(transform=ImageTransform(self.preprocessing, training))

    def build_trainer(self):
        return self.trainer_class(self.model, origin=self.origin)

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        if self.default_epochs is not None:
            settings = {"epochs": self.default_epochs, **settings}
        report = self.build_trainer().fit(
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
            **self.checkpoint_metadata(),
            "state_dict": {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            },
        }

    def checkpoint_metadata(self):
        return {
            "construction": self.construction,
            "preprocessing": self.preprocessing,
            "origin": self.origin,
        }

    def unpack_state(self, state):
        self.validate_state(state)
        return state["construction"], state["preprocessing"], state["origin"], state["state_dict"]

    @staticmethod
    def validate_state(state):
        if state.get("version") != 1:
            raise ValueError("Unsupported image task checkpoint version")
