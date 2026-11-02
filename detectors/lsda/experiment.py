"""LSDA Task composes native grouped data, mathematical model and trainer."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .checkpoint import initialize_b4
from .dataset import DEFAULT_LABEL_MAP, LSDADataset, LSDACollator
from .detector import LSDADetector
from .sampler import LSDAGroupSampler
from .trainer import LSDATrainer


RECIPE = {
    "resolution": 256,
    "domains": ["real", "DF", "F2F", "FS", "NT"],
    "frame_slots": 32,
    "fake_encoders_and_student": "native_original_B4_stem_unused1000head_fresh1792_to512",
    "content": "source_random_IResNet100_partial_freeze_stem_layer4_train_and_BN_train",
    "latent_augmentation": "literal_channelwise8x8_means_hard_examples_and_native_mixup",
    "objective": "domainCE+.5binaryCE+live_distillation+signed_real_cosine_constraint",
    "image_augmentation": "native_albu131_brightness_flip_nearest_down_linear_up_sharpen_JPEG_always_active",
    "sampling": "python_group_shuffle_one_shared_slot_per5videos_collect_batch_before_augmentation",
    "scheduler": None,
}


def build_model():
    fake = [load_backbone("lsda_b4", "vision", model_type="lsda_b4") for _ in range(4)]
    content = load_backbone("lsda_iresnet100", "vision", model_type="lsda_iresnet100")
    student = load_backbone("lsda_b4", "vision", model_type="lsda_b4")
    return LSDADetector(fake, content, student)


class LSDATask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        expected = preprocessing_config(256, (0.5,) * 3, (0.5,) * 3, augmentation=False)
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
                "specific_label_map": deepcopy(
                    settings.get("specific_label_map", DEFAULT_LABEL_MAP)
                ),
                "max_decode_attempts": settings.get("max_decode_attempts", 32),
            }
            preprocessing = expected
            origin = {
                **reference_for("lsda"),
                "selection_policy": "Remake explicit grouped train manifest and full-source validation, not native full experiment",
                "decode_repairs": "Bounded same-video neighboring attempts; explicit manifest replaces eager CUDA-name/path and cross-split/c40 discovery",
                "native_data_aug_flag_ignored": "Original LSDA getitem always applies its dedicated augmentation, despite config use_data_augmentation:false",
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
                    "LSDA requires original B4 init_weights or explicit initialization: random"
                )
        if (
            set(construction) != {"recipe", "specific_label_map", "max_decode_attempts"}
            or construction["recipe"] != RECIPE
            or preprocessing != expected
        ):
            raise ValueError(
                "LSDA requires the complete native default grouped model/data recipe"
            )
        label_map, attempts = (
            construction["specific_label_map"],
            construction["max_decode_attempts"],
        )
        if (
            not isinstance(label_map, dict)
            or any(
                not isinstance(key, str) or type(value) is not int or not 0 <= value < 5
                for key, value in label_map.items()
            )
            or type(attempts) is not int
            or attempts < 1
        ):
            raise ValueError(
                "LSDA requires a valid five-domain label map and positive max_decode_attempts"
            )
        if weights is not None:
            with torch.device("meta"):
                model = build_model()
            load_strict(model, weights, assign=True)
        else:
            model = build_model()
            if settings.get("init_weights"):
                origin["initialization"] = [
                    initialize_b4(encoder, settings["init_weights"])
                    for encoder in [
                        *model.model.encoders_f,
                        model.model.student_encoder,
                    ]
                ]
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
            LSDADataset(
                dataset,
                label_map=self.construction["specific_label_map"],
                max_decode_attempts=self.construction["max_decode_attempts"],
            )
            if stage == "train"
            else dataset
        )

    def get_collate_fn(self, stage):
        return LSDACollator() if stage == "train" else super().get_collate_fn(stage)

    def get_sampler(self, dataset, *, stage, batch_size, shuffle, seed):
        if stage != "train":
            return None
        if not isinstance(dataset, LSDADataset) or shuffle:
            raise ValueError(
                "LSDA requires grouped inventory and shuffle:false; its sampler owns group shuffling"
            )
        return LSDAGroupSampler(dataset.num_groups, batch_size=batch_size)

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = LSDATrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report


def create_task(settings, *, state=None):
    return LSDATask(settings, state=state)
