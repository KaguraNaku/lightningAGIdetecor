"""Task, constructor recipe and self-contained two-stage method state."""

from copy import deepcopy
from pathlib import Path

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.efficientnetb4.checkpoint import initialize_backbone

from .detector import MultiAttentionDetector
from .loss import MultiAttentionObjective
from .state import pack_state, restore_state
from .trainer import InferenceScorer, MultiAttentionTrainer


RECIPE = {
    "feature_layer": "b2",
    "attention_layer": "b5",
    "num_attentions": 4,
    "mid_dim": 256,
    "dropout": 0.25,
    "unused_dropout_final": 0.5,
    "constructor_bn_probe": True,
    "stem_head_swish": False,
    "drop_connect": False,
    "agda": {
        "kernel_size": 11,
        "sigma": 7.0,
        "unused_dilation": 2,
        "threshold": [0.4, 0.6],
        "zoom": [3, 5],
        "scale_factor": 0.5,
        "noise_rate": 0.1,
    },
    "ril": {
        "alpha": 0.05,
        "alpha_decay": 0.9,
        "inter_margin": 0.2,
        "intra_margin": [0.05, 0.1],
        "weights": [1, 1],
    },
    "scheduler": "step5_gamma0.5_once_after_fit_local_runner",
}


class MultiAttentionTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights, method = None, None
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
                raise ValueError(
                    "MultiAttention checkpoint lacks required method_state"
                )
        else:
            period, warmup = (
                settings.get("batch_per_epoch", 3591),
                settings.get("backbone_nEpochs", 10),
            )
            if (
                type(period) is not int
                or period < 1
                or type(warmup) is not int
                or warmup < 0
            ):
                raise ValueError(
                    "MultiAttention phase periods must be nonnegative epochs and positive batches"
                )
            construction = {
                "backbone": {
                    "num_classes": 2,
                    "in_channels": 3,
                    "mode": "Original",
                    "dropout": 0.0,
                },
                "image_size": 256,
                "batch_per_epoch": period,
                "backbone_nEpochs": warmup,
                "recipe": deepcopy(RECIPE),
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("multi_attention"),
                "objective": "native_warmup_ce_then_ce_plus_ril",
                "counter_semantics": "all_forward_calls_including_inference",
            }
            if settings.get("initial_checkpoint"):
                if not settings.get("initial_method_state"):
                    raise ValueError(
                        "Raw MultiAttention weights require an explicit initial_method_state sidecar; phase and RIL state cannot be inferred"
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
                    "Supply MultiAttention init_weights or explicitly select initialization: random"
                )
        period, warmup = (
            construction["batch_per_epoch"],
            construction["backbone_nEpochs"],
        )
        if (
            construction.get("recipe") != RECIPE
            or construction.get("image_size") != 256
            or construction.get("backbone")
            != {"num_classes": 2, "in_channels": 3, "mode": "Original", "dropout": 0.0}
            or type(period) is not int
            or period < 1
            or type(warmup) is not int
            or warmup < 0
        ):
            raise ValueError(
                "MultiAttention state must preserve native geometry, recipe and phase periods"
            )

        def backbone():
            return load_backbone(
                "efficientnetb4",
                "vision",
                model_type="efficientnetb4",
                **construction["backbone"],
            )

        if weights is not None:
            with torch.device("meta"):
                base = backbone()
                objective = MultiAttentionObjective(period)
                model = MultiAttentionDetector(
                    base, warmup_batches=period * warmup, probe=False
                )
            load_strict(model, weights, assign=True)
            scheduler_state = restore_state(model, objective, method, assign=True)
        else:
            base = backbone()
            if settings.get("init_weights"):
                origin["pretrained_backbone"] = initialize_backbone(
                    base, settings["init_weights"]
                )
            else:
                origin["initialization"] = "explicit_random"
            objective = MultiAttentionObjective(period)
            model = MultiAttentionDetector(base, warmup_batches=period * warmup)
            scheduler_state = None
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )
        self.objective, self.scheduler_state = objective, scheduler_state

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        trainer = MultiAttentionTrainer(
            self.model,
            self.objective,
            origin=self.origin,
            scheduler_state=self.scheduler_state,
        )
        report = trainer.fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self.scheduler_state, self._ready = trainer.scheduler_state, True
        return report

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import MultiAttention weights before prediction")
        self.model.to(runtime.device).eval()
        return InferenceScorer(self.model).predict(images, runtime=runtime)

    def method_state_dict(self):
        return pack_state(self.model, self.objective, self.scheduler_state)

    def state_dict(self):
        return {**super().state_dict(), "method_state": self.method_state_dict()}


def create_task(settings, *, state=None):
    return MultiAttentionTask(settings, state=state)
