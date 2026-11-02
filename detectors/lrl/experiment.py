"""LRL Task -> synchronized auxiliary inputs / native Trainer -> dual-B4 model."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.efficientnetb4.checkpoint import initialize_backbone
from .dataset import LRLDataset, LRLCollator
from .detector import LRLDetector
from .preprocessing import frequency_from_normalized
from .trainer import LRLTrainer, LRLScorer


CONSTRUCTION = {
    "backbone": "two_native_B4_static380_with_fresh_RGB3_then_IDCT3_replaced1_stems",
    "recipe": {
        "resolution": 256,
        "branch_dropout_unused": 0.3,
        "RFAM": "native_inplace_56_160_1792_sigmoid",
        "features": "2008x8x8_nearest_low.25_mid.5",
        "relation": "unfold2_stride2_cosine_noepsilon_16x16_drop_diagonal240",
        "classifier": "Linear240_128_ReLU_Dropout.2_Linear2",
        "decoder": "native_reverse_stages_InstanceNorm_nearest_then_bilinear_align_cornersTrue",
        "objective": "CE+sigmoid_mask_BCE+16x16_similarity_MSE_equal_weights",
        "IDCT": "native_RGBgray_float32_DCT_high_threshold.33_IDCT_then_div255_meanstd.5_no_clipping",
        "transport_repairs": [
            "named_5value_base_fields",
            "HW_HWC1_to_1HW",
            "float64_missing_zeros_to_float32_BCE",
        ],
        "decode_failure": "diagnostic_stop_without_skip_or_source_replacement",
        "scheduler": None,
    },
}


class LRLTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        expected_preprocessing = preprocessing_config(256, (0.5,) * 3, (0.5,) * 3)
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
        else:
            construction, preprocessing = deepcopy(CONSTRUCTION), expected_preprocessing
            origin = {
                **reference_for("lrl"),
                "model_math": "Unmodified native model reference with actual B4 ImageNet; mathematical parity is separate from broken full data entry",
                "data_transport": "Explicit tuple/rank/dtype repair; original failures retained, not a full original data-entry equivalence claim",
                "selection_policy": "Remake manifests/source validation; no full scientific reproduction claim",
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
                    "LRL requires original B4 init_weights or explicit initialization: random"
                )
        if construction != CONSTRUCTION or preprocessing != expected_preprocessing:
            raise ValueError(
                "LRL requires its complete native model/math and explicit data-repair recipe"
            )

        def build():
            options = {"model_type": "efficientnetb4", "dropout": 0.3}
            return LRLDetector(
                load_backbone("efficientnetb4", "vision", **options),
                load_backbone("efficientnetb4", "vision", **options),
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
                        ("RGB", model.encoder_rgb),
                        ("IDCT", model.encoder_idct),
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
        return LRLDataset(dataset, self.preprocessing, training=stage == "train")

    def get_collate_fn(self, stage):
        if stage not in ("train", "validation", "evaluation"):
            raise ValueError("Unknown LRL preprocessing stage")
        return LRLCollator()

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = LRLTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict_batch(self, batch, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import LRL weights before prediction")
        self.model.to(runtime.device).eval()
        return LRLScorer(self.model).predict_batch(batch, runtime=runtime)

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import LRL weights before prediction")
        return self.predict_batch(
            {"pixel_values": images, "idct": frequency_from_normalized(images)},
            runtime=runtime,
        )


def create_task(settings, *, state=None):
    return LRLTask(settings, state=state)
