"""UCFTask -> native paired Trainer -> common/specific reconstruction detector."""

from copy import deepcopy

import torch
from torch import nn

from detectors._shared.checkpoint import restore_model
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.checkpoint import read_initial_detector_weights
from .backbone import UCFXception
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


UFD_LABEL_MAP = {'real': 0, 'FF_Deepfakes': 1, 'FF_Face2Face': 2,
           'FF_FaceSwap': 3, 'FF_NeuralTextures': 4, 'ProGAN': 5}


def expand_progan_head(task):
    old = task.model.head_spe.mlp[2]
    if old.out_features != 5:
        raise ValueError('Expected five FF++ auxiliary classes before adding ProGAN')
    expanded = nn.Linear(old.in_features, 6, bias=old.bias is not None,
                         device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        expanded.weight[:5].copy_(old.weight)
        if old.bias is not None:
            expanded.bias[:5].copy_(old.bias)
    task.model.head_spe.mlp[2] = expanded
    task.construction['specific_classes'] = 6
    task.construction['specific_label_map'] = dict(UFD_LABEL_MAP)


class UCFTask(ImageClassificationTask):
    trainer_class = UCFTrainer

    @staticmethod
    def ufd_model_settings():
        return {"specific_classes": len(UFD_LABEL_MAP), "specific_label_map": dict(UFD_LABEL_MAP)}

    def configure_ufd_training(self, adaptation, *, warm_start):
        if warm_start:
            expand_progan_head(self)
        adaptation["UCF_domain_map"] = dict(UFD_LABEL_MAP)

    def __init__(self, settings, *, state=None):
        expected_preprocessing = preprocessing_config(256, (0.5,) * 3, (0.5,) * 3)
        if state is None:
            construction, origin = self._training_config(settings)
            preprocessing = expected_preprocessing
            weights, receipt = read_initial_detector_weights(settings, detector="UCF")
            origin.update(receipt)
        else:
            construction, preprocessing, origin, weights = self.unpack_state(state)

        self._validate_recipe(construction, preprocessing, expected_preprocessing)

        def build():
            return UCFDetector(
                UCFXception(), UCFXception(), specific_classes=construction["specific_classes"]
            )

        if weights is not None:
            model = restore_model(build, weights)
        else:
            model = build()
            self._initialize_encoders(model, settings.get("init_weights"), origin)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    @staticmethod
    def _training_config(settings):
        construction = {
            "recipe": deepcopy(RECIPE),
            "specific_classes": settings.get("specific_classes", 5),
            "specific_label_map": deepcopy(settings.get("specific_label_map", DEFAULT_LABEL_MAP)),
        }
        origin = {
            "method": "ucf",
            "selection_policy": "Explicit source manifests and validation",
            "label_binding": "Explicit metadata overrides declared category map; no automatic alias correction or type renumbering",
            "evaluation_observers": "Native label/probability counters optional in direct model API; unified evaluator owns metrics",
            "exact_optimizer_rng_resume_supported": False,
        }
        return construction, origin

    @staticmethod
    def _validate_recipe(construction, preprocessing, expected_preprocessing):
        if (
            set(construction) != {"recipe", "specific_classes", "specific_label_map"}
            or construction["recipe"] != RECIPE
            or preprocessing != expected_preprocessing
        ):
            raise ValueError("UCF requires complete native model, loss and pair-transport recipe")
        count = construction["specific_classes"]
        label_map = construction["specific_label_map"]
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
            raise ValueError("UCF requires valid specific classes and an explicit source label map")

    @staticmethod
    def _initialize_encoders(model, init_weights, origin):
        if init_weights:
            origin["initialization"] = {
                "forgery": initialize_backbone(model.encoder_f, init_weights),
                "content": initialize_backbone(model.encoder_c, init_weights),
            }
        else:
            origin["initialization"] = "explicit_constructed_random"

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

    def build_collator(self, *, training):
        return UCFCollator() if training else super().build_collator(training=training)
