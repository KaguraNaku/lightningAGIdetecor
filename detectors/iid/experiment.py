"""IIDTask -> identity-aware Trainer -> implicit/explicit identity detector."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .checkpoint import initialize_backbone, initialize_explicit
from .dataset import IIDDataset, IIDCollator
from .detector import IIDDetector
from .trainer import IIDTrainer


CONSTRUCTION = {
    "backbone": "iid_adjust_channel_xception_plus_native_iresnet50",
    "recipe": {
        "resolution": 256,
        "identity_resolution": 112,
        "embedding": 512,
        "classifier": "Linear(implicit512-explicit512)->2",
        "identity_classes": 1000,
        "objective": "CE8_sigmoid_margin.2+.05*ArcFace64_margin.4+.1*(cos_label1-cos_label0)",
        "explicit_network": "trainable_except_final_features.weight; initially_eval_then_parent_mode",
        "explicit_final_BN": "only_if_batchsize_greater_than1_even_in_eval",
        "identity_transport": "explicit_metadata_or_native_parent_last_underscore_integer0..999",
        "decode_failure": "first_source_actual_identity_and_label_once_then_fatal",
        "norm": "literal_no_epsilon",
        "scheduler": None,
    },
}


class IIDTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
        else:
            construction = deepcopy(CONSTRUCTION)
            preprocessing = preprocessing_config(
                256, (0.5,) * 3, (0.5,) * 3, augmentation=True
            )
            origin = {
                **reference_for("iid"),
                "recognition_weights_role": "Trainable explicit identity branch; never a frozen teacher",
                "selection_policy": "Explicit manifests/source validation; not original full experiment",
                "failure_boundary": "One native index0 source/label/identity replacement; repeated decode or nonfinite objective fatal",
                "exact_optimizer_rng_resume_supported": False,
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif settings.get("initialization") != "random" and not (
                settings.get("init_weights") and settings.get("explicit_weights")
            ):
                raise ValueError(
                    "IID requires Xception init_weights AND explicit_weights, or explicit initialization: random"
                )
        if construction != CONSTRUCTION or preprocessing != preprocessing_config(
            256, (0.5,) * 3, (0.5,) * 3, augmentation=True
        ):
            raise ValueError(
                "IID requires complete fixed native identity/model/loss/preprocessing recipe"
            )

        def build():
            return IIDDetector(
                load_backbone("iid_xception", "vision", model_type="iid_xception"),
                load_backbone("iresnet50_iid", "vision", model_type="iresnet50_iid"),
            )

        if weights is not None:
            with torch.device("meta"):
                model = build()
            load_strict(model, weights, assign=True)
        else:
            model = build()
            if settings.get("init_weights"):
                origin["vision_initialization"] = initialize_backbone(
                    model.backbone, settings["init_weights"]
                )
            if settings.get("explicit_weights"):
                origin["recognition_initialization"] = initialize_explicit(
                    model.explicit_extractor, settings["explicit_weights"]
                )
            origin["initialization"] = (
                "explicit_constructed_random_allowed"
                if settings.get("initialization") == "random"
                else "supplied_two_branch_initialization"
            )
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def prepare_dataset(self, dataset, *, stage):
        return IIDDataset(dataset, self.preprocessing) if stage == "train" else dataset

    def get_collate_fn(self, stage):
        return IIDCollator() if stage == "train" else super().get_collate_fn(stage)

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        result = IIDTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return result


def create_task(settings, *, state=None):
    return IIDTask(settings, state=state)
