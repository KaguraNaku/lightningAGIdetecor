"""Native UIA task with full banks/statistics and explicit labeled batch route."""

from copy import deepcopy
from pathlib import Path

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .detector import UIAViTDetector
from .trainer import LabeledScorer, UIAObjective, UIATrainer


RECIPE = {
    "image_size": 224,
    "patch_size": 16,
    "dimension": 768,
    "depth": 12,
    "heads": 12,
    "feature_before_block": 6,
    "previous_attention_indices": [8, 9, 10, 11, 12],
    "progressive": {"uniform_until": 0.125, "k": 12, "threshold": 0.7},
    "loss_weights": [0.06, 0.05, 0.5],
    "fake_patch_slice": [3, 11],
    "collect_during_evaluation": True,
    "bank_policy": "all_samples_ordered_lossless_disk_chunks",
}


def initialize_backbone(model, path):
    weights, receipt = read_weights(path)
    adapted = {
        key: value for key, value in weights.items() if not key.startswith("head.")
    }
    expected = model.state_dict()
    allowed = {
        "head.weight",
        "head.bias",
        "norm_middle.weight",
        "norm_middle.bias",
        "c_real",
        "c_fake",
        "c_cross",
    }
    missing, unexpected = set(expected) - set(adapted), set(adapted) - set(expected)
    if unexpected or missing - allowed:
        raise ValueError(
            f"UIA ImageNet layout differs: missing={sorted(missing - allowed)}, unexpected={sorted(unexpected)}"
        )
    result = model.load_state_dict(adapted, strict=False)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("UIA initialization key accounting mismatch")
    receipt.update(
        ignored_head_keys=sorted(key for key in weights if key.startswith("head.")),
        initialized_keys=sorted(adapted),
        retained_native_initialization_keys=sorted(missing),
    )
    return receipt


class UIAViTTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = method = None
        bank_directory = settings.get("bank_directory")
        if state is not None and bank_directory and not Path(bank_directory).is_dir():
            bank_directory = None  # Serialized banks do not depend on old cache paths.
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
            method = state.get("method_state")
            if method is None:
                raise ValueError("UIA checkpoint requires complete method_state")
        else:
            construction = {
                "recipe": deepcopy(RECIPE),
                "batch_per_epoch": settings.get("batch_per_epoch", 1796),
                "nEpochs": settings.get("nEpochs", 10),
            }
            preprocessing = preprocessing_config(
                224,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("uia_vit"),
                "objective": "native_ce_plus_patch_consistency",
                "labeled_evaluation": "collect_features_and_advance_counter",
                "label_free_scores": "read_only_model_evaluation_step1",
            }
            if settings.get("initial_checkpoint"):
                if not settings.get("initial_method_state"):
                    raise ValueError(
                        "Raw UIA weights require an explicit initial_method_state sidecar"
                    )
                weights, receipt = read_weights(settings["initial_checkpoint"])
                sidecar = Path(settings["initial_method_state"]).expanduser().resolve()
                method = torch.load(sidecar, map_location="cpu", weights_only=True)
                origin.update(receipt)
                from data.manifest import sha256

                origin["method_state_source"] = {
                    "path": str(sidecar),
                    "sha256": sha256(sidecar),
                }
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply UIA init_weights or explicitly select initialization: random"
                )
        if construction.get("recipe") != RECIPE:
            raise ValueError("UIA state must preserve the native B/16 recipe")

        def backbone():
            return load_backbone("uia_vit", "vision", model_type="uia_vit")

        if weights is not None:
            with torch.device("meta"):
                model = UIAViTDetector(
                    backbone(),
                    batch_per_epoch=construction["batch_per_epoch"],
                    epochs=construction["nEpochs"],
                    bank_directory=bank_directory,
                )
            load_strict(model, weights, assign=True)
            model.load_method_state_dict(method)
        else:
            core = backbone()
            if settings.get("init_weights"):
                origin["pretrained_backbone"] = initialize_backbone(
                    core, settings["init_weights"]
                )
            else:
                origin["initialization"] = "explicit_random"
            model = UIAViTDetector(
                core,
                batch_per_epoch=construction["batch_per_epoch"],
                epochs=construction["nEpochs"],
                bank_directory=bank_directory,
            )
        # Bind PCL to the materialized model Parameters after assign=True loading.
        objective = UIAObjective(model.model)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )
        self.objective = objective

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = UIATrainer(self.model, self.objective, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 11,
                "learning_rate": 3e-5,
                "weight_decay": 1e-5,
                **settings,
            },
            output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict_batch(self, batch, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import UIA weights before prediction")
        self.model.to(runtime.device).eval()
        return LabeledScorer(self.model).predict_batch(batch, runtime=runtime)

    def state_dict(self):
        return {**super().state_dict(), "method_state": self.model.method_state_dict()}


def create_task(settings, *, state=None):
    return UIAViTTask(settings, state=state)
