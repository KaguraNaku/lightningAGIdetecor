"""Statistical Hybrid ZCA Task on the shared remake data/evaluation interfaces."""

import itertools
import json
from pathlib import Path

import torch
from transformers import AutoConfig, CLIPVisionConfig, CLIPVisionModel

from backbones.load_backbone import load_backbone
from data.manifest import sha256
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from engine.runtime import atomic_save
from .calibration import HybridZCACalibrator
from .checkpoint import translate_native
from .data import HybridPairCollator, HybridPairDataset, HybridPairSampler
from .detector import HybridZCADetector

MODEL_ID = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class HybridZCATask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
            )
            if (
                construction.get("method") != "hybrid_zca"
                or construction.get("prediction_unit") != "image"
            ):
                raise ValueError("Not a canonical Hybrid ZCA image checkpoint")
            cfg = CLIPVisionConfig.from_dict(construction["vision_config"])
            weights = state["state_dict"]
            native = False
        else:
            if settings.get("initialization") is not None:
                raise ValueError(
                    "Hybrid ZCA requires pretrained CLIP or a calibrated native checkpoint"
                )
            attention = settings.get("attention_implementation", "sdpa")
            if attention not in ("sdpa", "eager"):
                raise ValueError("Hybrid ZCA attention must be sdpa or eager")
            model_id = settings.get("backbone_id", MODEL_ID)
            options = {
                "revision": settings.get("revision", REVISION),
                "local_files_only": settings.get("local_files_only", True),
            }
            origin = {
                **reference_for("hybrid_zca"),
                "objective": "source-label statistical calibration; no optimizer",
                "pretrained_vision": {"model_id": model_id, **options},
            }
            native = settings.get("initial_checkpoint") is not None
            if native:
                if settings.get("checkpoint_calibrated") is not True:
                    raise ValueError(
                        "Native tensors do not record readiness: require explicit checkpoint_calibrated:true"
                    )
                cfg = AutoConfig.from_pretrained(model_id, **options)
                cfg = getattr(cfg, "vision_config", cfg)
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            else:
                vision = load_backbone(
                    model_id,
                    "vision",
                    model_type="clip",
                    attn_implementation=attention,
                    **options,
                )
                cfg = vision.config
            construction = {
                "method": "hybrid_zca",
                "prediction_unit": "image",
                "vision_config": cfg.to_dict(),
                "attention_implementation": attention,
                "layers": settings.get("layers", [12, 18]),
                "memory_size": settings.get("memory_size", 2048),
                "pair_shuffle": settings.get("pair_shuffle", True),
                "sampler_profile": "seed-explicit pair randperm, complete batches only; conditioned ordering, not original global worker RNG replay",
            }
            if type(construction["pair_shuffle"]) is not bool:
                raise ValueError("pair_shuffle must be boolean")
            preprocessing = preprocessing_config(224, MEAN, STD, augmentation=False)
        if preprocessing != preprocessing_config(224, MEAN, STD, augmentation=False):
            raise ValueError(
                "Hybrid ZCA checkpoint preprocessing differs from native profile"
            )
        if construction["attention_implementation"] not in ("sdpa", "eager"):
            raise ValueError("Invalid saved Hybrid ZCA attention implementation")
        if (
            cfg.intermediate_size != 4096
            or cfg.hidden_act != "quick_gelu"
            or cfg.layer_norm_eps != 1e-5
            or cfg.attention_dropout != 0.0
        ):
            raise ValueError(
                "Hybrid ZCA requires native CLIP-L14 MLP/activation/norm/dropout"
            )
        if weights is not None and any(
            value.dtype != torch.float32 or not torch.isfinite(value).all()
            for value in weights.values()
        ):
            raise ValueError("Hybrid ZCA checkpoint requires finite float32 tensors")
        cfg._attn_implementation = construction["attention_implementation"]
        if weights is not None:
            with torch.device("meta"):
                model = HybridZCADetector(
                    CLIPVisionModel(cfg),
                    layers=construction["layers"],
                    memory_size=construction["memory_size"],
                )
            if native:
                weights, receipt = translate_native(model, weights)
                origin["native_translation"] = receipt
            load_strict(model, weights, assign=True)
        else:
            model = HybridZCADetector(
                vision,
                layers=construction["layers"],
                memory_size=construction["memory_size"],
            )
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def prepare_dataset(self, dataset, *, stage):
        return HybridPairDataset(dataset) if stage == "train" else dataset

    def get_collate_fn(self, stage):
        return (
            HybridPairCollator(self.preprocessing)
            if stage == "train"
            else super().get_collate_fn(stage)
        )

    def get_sampler(self, dataset, *, stage, batch_size, shuffle, seed):
        if stage != "train":
            return None
        if not isinstance(dataset, HybridPairDataset) or shuffle:
            raise ValueError(
                "Hybrid ZCA requires paired data and loader shuffle:false; pair_shuffle owns permutation"
            )
        return HybridPairSampler(
            dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=self.construction["pair_shuffle"],
        )

    @staticmethod
    def _check_runtime(runtime):
        if runtime.precision != "fp32":
            raise ValueError(
                "Hybrid ZCA currently requires fp32; AMP parity is unverified"
            )

    @torch.no_grad()
    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._check_runtime(runtime)
        if validation_loader is not None:
            raise ValueError("Hybrid ZCA has no validation selection or optimizer")
        if not isinstance(loader.dataset, HybridPairDataset) or not isinstance(
            loader.sampler, HybridPairSampler
        ):
            raise ValueError(
                "Hybrid ZCA fit requires the method-owned pair dataset and sampler"
            )
        allowed = {
            "input",
            "calibration_batches",
            "feature_budget_mib",
            "patch_budget_gib",
            "disk_reserve_gib",
        }
        if set(settings) - allowed:
            raise ValueError(
                f"Unsupported Hybrid ZCA fit settings: {sorted(set(settings) - allowed)}"
            )
        limit = settings.get("calibration_batches", 100)
        if type(limit) is not int or limit < 1:
            raise ValueError("calibration_batches must be positive")
        self._ready = False
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory = loader.dataset.inventory()
        inventory["manifest"] = str(loader.dataset.manifest_path)
        inventory["manifest_sha256"] = sha256(loader.dataset.manifest_path)
        inventory["sampler_profile"] = self.construction["sampler_profile"]
        atomic_save(output_dir / "pair-inventory.json", inventory)
        self.model.to(runtime.device).eval()
        fitter = HybridZCACalibrator(
            self.model.statistics,
            output_dir / "calibration",
            **{
                k: settings[k]
                for k in ("feature_budget_mib", "patch_budget_gib", "disk_reserve_gib")
                if k in settings
            },
        )
        selected = []
        batches = 0
        try:
            with (output_dir / "calibration-samples.jsonl").open("x") as stream:
                for batch in itertools.islice(loader, limit):
                    n = len(batch["pair_indices"])
                    if (
                        n != loader.batch_size
                        or batch["labels"].tolist() != [0] * n + [1] * n
                    ):
                        raise ValueError(
                            "Hybrid ZCA requires complete real-then-fake native pair batches"
                        )
                    selected.extend(batch["pair_indices"])
                    pixels = batch["pixel_values"].to(runtime.device)
                    real, patches = self.model.features(
                        pixels[:n], collect_real_patches=True
                    )
                    fake, _ = self.model.features(pixels[n:])
                    fitter.add(real, fake, patches)
                    for index, metadata in enumerate(batch["metadata"]):
                        stream.write(
                            json.dumps(
                                {
                                    "batch": batches,
                                    "pair_index": batch["pair_indices"][index % n],
                                    "side": "real" if index < n else "fake",
                                    **metadata,
                                }
                            )
                            + "\n"
                        )
                    batches += 1
            report = fitter.finish()
        finally:
            fitter.close()
        indices = loader.sampler.indices
        if indices is None:
            raise ValueError("Hybrid ZCA sampler was not consumed")
        selected_set = set(selected)
        report.update(
            operation="source-only frozen calibration",
            optimizer=None,
            batches=batches,
            calibration_batch_limit=limit,
            selected_pair_indices=selected,
            unselected_pair_indices=[
                i for i in range(len(loader.dataset)) if i not in selected_set
            ],
            dropped_tail_pair_indices=indices[len(loader.sampler) :],
            source_manifest_sha256=inventory["manifest_sha256"],
            pair_inventory=str(output_dir / "pair-inventory.json"),
            source_samples=str(output_dir / "calibration-samples.jsonl"),
            sampler_profile=inventory["sampler_profile"],
        )
        atomic_save(output_dir / "fit-report.json", report)
        self.origin["calibration"] = report
        self._ready = True
        return report

    def predict(self, images, *, runtime):
        self._check_runtime(runtime)
        return super().predict(images, runtime=runtime)


def create_task(settings, *, state=None):
    return HybridZCATask(settings, state=state)
