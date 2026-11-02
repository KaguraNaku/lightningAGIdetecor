"""UCFTask -> native paired Trainer -> common/specific reconstruction detector."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .checkpoint import initialize_backbone
from .dataset import UCFDataset, UCFCollator, DEFAULT_LABEL_MAP
from .detector import UCFDetector
from .trainer import UCFTrainer


RECIPE = {
    "resolution": 256,
    "feature_dim": 512,
    "half_feature_dim": 256,
    "backbone": "native_adjust_channel_Xception_unused2048_classifier",
    "reconstruction": "shared_native_AdaIN_unbiased1e-5_UNet_4ordered_calls_dropout.3",
    "shuffle": "native_inplace_within_halves_python_p.7",
    "objective": "commonCE+.1*specificCE+.3*fourL1+.05*contrastive_margin3_default_pairwise_epsilon",
    "pairing": "fake_indexed_real_sample_with_replacement_before_IO;fake_then_real_augmentation;all_real_then_all_fake",
    "decode_failure": "native_fatal_no_replacement_drop_or_relabel",
    "scheduler": None,
}


class UCFTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        expected = preprocessing_config(256, (0.5,) * 3, (0.5,) * 3)
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
        else:
            construction = {
                "recipe": deepcopy(RECIPE),
                "specific_classes": settings.get("specific_classes", 5),
                "specific_label_map": deepcopy(
                    settings.get("specific_label_map", DEFAULT_LABEL_MAP)
                ),
            }
            preprocessing = expected
            origin = {
                **reference_for("ucf"),
                "selection_policy": "Remake explicit source manifests and validation; not native full experiment",
                "label_binding": "Explicit metadata overrides declared category map; no automatic alias correction or type renumbering",
                "evaluation_observers": "Native label/probability counters optional in direct model API; unified evaluator owns metrics",
                "exact_optimizer_rng_resume_supported": False,
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "UCF requires original Xception init_weights or explicit initialization: random"
                )
        if (
            set(construction) != {"recipe", "specific_classes", "specific_label_map"}
            or construction["recipe"] != RECIPE
            or preprocessing != expected
        ):
            raise ValueError(
                "UCF requires complete native model, loss and pair-transport recipe"
            )
        count, label_map = (
            construction["specific_classes"],
            construction["specific_label_map"],
        )
        if (
            type(count) is not int
            or count < 2
            or not isinstance(label_map, dict)
            or any(
                not isinstance(key, str)
                or type(value) is not int
                or not 0 <= value < count
                for key, value in label_map.items()
            )
        ):
            raise ValueError(
                "UCF requires valid specific classes and an explicit source label map"
            )

        def build():
            return UCFDetector(
                load_backbone("ucf_xception", "vision", model_type="ucf_xception"),
                load_backbone("ucf_xception", "vision", model_type="ucf_xception"),
                specific_classes=count,
            )

        if weights is not None:
            with torch.device("meta"):
                model = build()
            load_strict(model, weights, assign=True)
        else:
            model = build()
            if settings.get("init_weights"):
                origin["initialization"] = {
                    name: initialize_backbone(encoder, settings["init_weights"])
                    for name, encoder in (
                        ("forgery", model.encoder_f),
                        ("content", model.encoder_c),
                    )
                }
            else:
                origin["initialization"] = "explicit_constructed_random"
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def prepare_dataset(self, dataset, *, stage):
        return (
            UCFDataset(
                dataset,
                self.preprocessing,
                specific_classes=self.construction["specific_classes"],
                label_map=self.construction["specific_label_map"],
            )
            if stage == "train"
            else dataset
        )

    def get_collate_fn(self, stage):
        return UCFCollator() if stage == "train" else super().get_collate_fn(stage)

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = UCFTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report


def create_task(settings, *, state=None):
    return UCFTask(settings, state=state)
