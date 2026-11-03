"""AltFreezing Task composition with explicit counter and gradient-flag state."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors.i3d.checkpoint import initialize_backbone
from detectors.i3d.dataset import I3DCollator, I3DDataset
from detectors.i3d.experiment import PREPROCESSING, RECIPE, trained_weights
from .detector import AltFreezingDetector
from .trainer import AltFreezingTrainer

ALT_RECIPE = {
    **RECIPE,
    "method": "altfreezing",
    "freeze_schedule": "post-forward spatial atcount%21==0,temporal atcount%21==1; counter increments only when inferenceFalse",
}


def build_model():
    return AltFreezingDetector(
        load_backbone("i3d_resnet50", "vision", model_type="i3d_resnet50")
    )


class AltFreezingTask(ImageClassificationTask):
    prediction_unit = "clip"

    def __init__(self, settings, *, state=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Native AltFreezing requires default float32 tensors")
        weights = schedule = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights, schedule = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
                state["schedule"],
            )
            if any(
                value.is_floating_point() and value.dtype != torch.float32
                for value in weights.values()
            ):
                raise ValueError(
                    "Saved AltFreezing floating weights must remain float32"
                )
        else:
            construction = {
                "recipe": deepcopy(ALT_RECIPE),
                "prediction_unit": "clip",
                "selection_seed": settings.get("selection_seed", 1024),
            }
            preprocessing = deepcopy(PREPROCESSING)
            origin = {
                **reference_for("altfreezing"),
                "selection_policy": "Explicit source train/val; no target-set checkpoint selection",
                "native_process_RNG_reproduced": False,
                "exact_optimizer_rng_resume_supported": False,
                "native_generic_Trainer_use_amp_default": True,
                "GPU_AMP_execution_verified": False,
                "full_experiment_reproduction": False,
                "legacy_import_schedule": "Original state_dict omits counter/flags; native legacy import starts count0/alltrainable like native construction. Canonical task preserves both.",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = trained_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "AltFreezing needs init_weights or explicit initialization:random"
                )
        if (
            set(construction) != {"recipe", "prediction_unit", "selection_seed"}
            or construction["recipe"] != ALT_RECIPE
            or construction["prediction_unit"] != "clip"
            or preprocessing != PREPROCESSING
            or type(construction["selection_seed"]) is not int
            or not 0 <= construction["selection_seed"] < 2**32
        ):
            raise ValueError(
                "AltFreezing needs the complete native clip/freeze recipe and seed"
            )
        if weights is not None:
            with torch.device("meta"):
                model = build_model()
            load_strict(model, weights, assign=True)
        else:
            model = build_model()
            origin["initialization"] = (
                initialize_backbone(model.resnet, settings["init_weights"])
                if settings.get("init_weights")
                else "explicit_constructed_random"
            )
        if schedule is not None:
            model.load_schedule_state_dict(schedule)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def get_collate_fn(self, stage):
        if stage not in ("train", "validation", "evaluation"):
            raise ValueError("Unknown AltFreezing preprocessing stage")
        return I3DCollator()

    def prepare_dataset(self, dataset, *, stage):
        return I3DDataset(
            dataset, stage=stage, seed=self.construction["selection_seed"]
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        report = AltFreezingTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict(self, clips, *, runtime):
        if not self._ready:
            raise ValueError(
                "Fit or import trained AltFreezing weights before prediction"
            )
        if runtime.precision != "fp32":
            raise ValueError("The selected AltFreezing alignment profile requires fp32")
        self.model.to(runtime.device).eval()
        scores = self.model(
            clips.to(runtime.device, non_blocking=True), inference=True
        ).view(-1)
        if (
            scores.shape != (len(clips),)
            or not torch.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise ValueError("AltFreezing must return one finite probability per clip")
        return scores

    def state_dict(self):
        state = super().state_dict()
        state["schedule"] = self.model.schedule_state_dict()
        return state


def create_task(settings, *, state=None):
    return AltFreezingTask(settings, state=state)
