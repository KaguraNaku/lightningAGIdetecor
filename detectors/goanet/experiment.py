"""Canonical GOANet task with explicit variant and scoped prior ownership."""

from importlib.resources import files
from copy import deepcopy
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch

from backbones.load_backbone import load_backbone
from data.manifest import sha256
from detectors._shared.checkpoint import read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .detector import GOANetDetector
from .trainer import GOANetTrainer
from .checkpoint import restore_weights


RECIPE = {
    "variant": "canonical_prior_guided",
    "tail_rank_cap": 32,
    "prior_dim": 64,
    "cdc_theta": 0.7,
    "cdc_output": "normal_plus_difference",
    "prior_filter": "native_signed_2x2_no_padding",
    "hard_projection": "U_res_minus_U_pri_U_pri_T_U_res",
    "gate": "1_plus_half_tanh_zero_initialized",
    "orthogonal_weight": 0.01,
    "consistency_loss_active": False,
    "scheduler": {
        "type": "cosine",
        "T_max": 10,
        "eta_min": 1e-5,
        "placement": "once_after_complete_fit",
        "native_warmup_epoch": 1,
    },
}


def default_specification():
    return json.loads(
        files("configs").joinpath("migrations/goanet_clip_l14.json").read_text()
    )


class GOANetTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = initial = None
        self.scheduler_state = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
            method = state.get("method_state")
            if not isinstance(method, dict) or set(method) != {"scheduler"}:
                raise ValueError("GOANet checkpoint requires scheduler method_state")
            self.scheduler_state = method["scheduler"]
        else:
            if (
                settings.get("source_variant", "canonical_prior_guided")
                != "canonical_prior_guided"
            ):
                raise ValueError(
                    "Only canonical prior-guided GOANet is connected; fixed is a different incompatible source variant"
                )
            spec = default_specification()
            construction = {
                "recipe": deepcopy(RECIPE),
                "vision_config": deepcopy(
                    settings.get("vision_config", spec["vision_config"])
                ),
                "attention_implementation": settings.get(
                    "attention_implementation", "sdpa"
                ),
            }
            origin = {
                **reference_for("goanet"),
                "source_variant": "goanet_detector.py",
                "objective": "CE_plus_point01_sum_principal_tail_Gram",
                "prior_lifecycle": "model_owned_scoped_context; transient last graph not serialized",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif settings.get("initialization") == "random":
                origin["initialization"] = "explicit_constructed_random"
            else:
                if settings.get("vision_config") is not None:
                    raise ValueError(
                        "Custom GOANet geometry requires random initialization or raw checkpoint"
                    )
                initial = (
                    Path(
                        settings.get("init_weights")
                        or hf_hub_download(
                            spec["model_id"],
                            "model.safetensors",
                            revision=spec["revision"],
                            local_files_only=True,
                        )
                    )
                    .expanduser()
                    .resolve()
                )
                origin["pretrained_vision"] = {
                    "path": str(initial),
                    "sha256": sha256(initial),
                    "model_id": spec["model_id"],
                    "revision": spec["revision"],
                }
            preprocessing = preprocessing_config(
                construction["vision_config"]["image_size"],
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
        if construction["recipe"] != RECIPE:
            raise ValueError("GOANet state must retain the canonical native recipe")
        if construction["attention_implementation"] not in ("sdpa", "eager"):
            raise ValueError("GOANet attention must be sdpa or eager")
        if self.scheduler_state is not None and not isinstance(
            self.scheduler_state, dict
        ):
            raise ValueError("GOANet scheduler state must be a dictionary or None")
        if self.scheduler_state is not None and (
            set(self.scheduler_state) != {"warmup_epoch", "scheduler"}
            or type(self.scheduler_state["warmup_epoch"]) is not int
            or self.scheduler_state["warmup_epoch"] < 0
            or not isinstance(self.scheduler_state["scheduler"], dict)
        ):
            raise ValueError("GOANet requires complete warmup/cosine scheduler state")

        def backbone():
            return load_backbone(
                "clip_checkpoint",
                "vision",
                model_type="clip_checkpoint",
                vision_config=construction["vision_config"],
                attention_implementation=construction["attention_implementation"],
                init_weights=initial,
            )

        if weights is not None:
            with torch.device("meta"):
                model = GOANetDetector(backbone)
            restore_weights(model, weights)
        else:
            model = GOANetDetector(backbone)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        trainer = GOANetTrainer(
            self.model, origin=self.origin, scheduler_state=self.scheduler_state
        )
        report = trainer.fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self.scheduler_state = trainer.scheduler_state
        self._ready = True
        return report

    def state_dict(self):
        return {
            **super().state_dict(),
            "method_state": {"scheduler": self.scheduler_state},
        }


def create_task(settings, *, state=None):
    return GOANetTask(settings, state=state)
