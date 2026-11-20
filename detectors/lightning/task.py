"""Shared-engine Task for the formal fixed-eight Lightning protocol.

The frozen encoder is a pinned shared resource. Checkpoints contain only the
small readout head, fixed mapping, sufficient statistics and processor identity;
they never duplicate encoder weights or persist per-image feature matrices.
"""
from pathlib import Path
import math
import os

import numpy as np
import torch

from data.manifest import sha256
from models.clip import load_clip_vision

from .detector import LightningDetector
from .encoder import CLIP_MODEL_ID, CLIP_WIDTH, CLIPEncoder
from .model import (
    BRANCHES, DEFAULT_RIDGE_LAMBDA, LAYERS, PROJECTION_RANK,
    AnalyticReadout, MappingAccumulator, SpectralMapping, SufficientStatistics,
)


_CHECKPOINT_HEADER = {"format": "fixed8-cls-mean", "version": 1}


class LightningTask:
    def __init__(self, settings, *, state=None):
        self.chain_id = settings.get("chain_id")
        self.stage_index = settings.get("stage_index", 0)
        self.backbone = self.detector = self.processor = None
        self._processor_state = None
        self._device = None
        self.model = None
        self.training_ids = set()

        self.encoder_spec = {"model_id": CLIP_MODEL_ID}
        if "local_files_only" in settings:
            self.encoder_spec["local_files_only"] = settings["local_files_only"]
        if settings.get("cache_dir"):
            self.encoder_spec["cache_dir"] = str(Path(settings["cache_dir"]).expanduser())
        self.ridge_lambda = settings.get("ridge_lambda", DEFAULT_RIDGE_LAMBDA)
        if state is not None:
            self._restore_state(state)
        elif settings.get("parent_checkpoint"):
            self._load_parent(settings)
        else:
            if self.chain_id is not None or self.stage_index != 0:
                raise ValueError("Set a parent checkpoint for a continuation stage")
            if "mapping_path" in settings:
                self.model = AnalyticReadout.from_mapping_file(self._asset(settings["mapping_path"]))

        if self.encoder_spec["model_id"] != CLIP_MODEL_ID:
            raise ValueError("Lightning uses the fixed CLIP-L/14 encoder")
        value = self.ridge_lambda
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Lightning requires a positive ridge_lambda")
        if self.model is not None:
            self._validate_mapping(self.model)

    @property
    def regularization(self):
        return {"kind": "uniform_mean_loss", "ridge_lambda": self.ridge_lambda}

    def _validate_mapping(self, model):
        expected = (len(LAYERS), CLIP_WIDTH, PROJECTION_RANK)
        if any(mapping.basis.shape != expected for mapping in model.mappings.values()):
            raise ValueError(f"Both Lightning maps must have basis shape {expected}")

    def _restore_state(self, state):
        if (any(state.get(key) != value for key, value in _CHECKPOINT_HEADER.items())
                or tuple(state.get("layers", ())) != LAYERS):
            raise ValueError("Unsupported fixed-eight Lightning checkpoint")
        regularization = state.get("regularization", {})
        if regularization.get("kind") != "uniform_mean_loss":
            raise ValueError("Lightning checkpoints must declare uniform_mean_loss")
        self.ridge_lambda = regularization.get("ridge_lambda")
        self.encoder_spec = state["encoder"]
        self._processor_state = state["processor"]
        ids = state["training_ids"]
        self.training_ids = set(ids)
        mappings = {b: SpectralMapping(**self._numpy(state["mapping"][b])) for b in BRANCHES}
        statistics = {b: SufficientStatistics.from_state(self._numpy(s))
                      for b, s in state.get("statistics", {}).items()}
        coefficients = self._numpy(state["coefficients"])
        if set(statistics) != set(BRANCHES) or set(coefficients) != {*BRANCHES, "bias"}:
            raise ValueError("A fitted Lightning checkpoint must contain both complete readout branches")
        width = len(LAYERS) * PROJECTION_RANK
        if (len(ids) != len(self.training_ids)
                or any(s.n != len(ids) or not 0 < s.sy < s.n
                       or s.g.shape != (width, width) or s.h.shape != (width,) or s.sz.shape != (width,)
                       or not all(np.isfinite(a).all() for a in (s.g, s.h, s.sz))
                       for s in statistics.values())):
            raise ValueError("Lightning source statistics do not match its training IDs or projected width")
        if (any(np.shape(coefficients[b]) != (len(LAYERS), CLIP_WIDTH) for b in BRANCHES)
                or np.shape(coefficients["bias"]) != ()
                or not all(np.isfinite(a).all() for a in coefficients.values())):
            raise ValueError("Invalid Lightning coefficients")
        self.model = AnalyticReadout(mappings, statistics, coefficients)

    def _load_parent(self, settings):
        if ("mapping_path" in settings or not isinstance(self.chain_id, str) or not self.chain_id.strip()
                or type(self.stage_index) is not int or self.stage_index < 1):
            raise ValueError("Lightning continuation requires chain_id and positive stage_index")
        from engine.fit import restore

        reference = settings["parent_checkpoint"]
        parent, saved = restore(self._asset(reference))
        if not isinstance(parent, type(self)):
            raise ValueError("Lightning continuation requires a fitted Lightning parent")
        if self.stage_index == 1:
            if parent.stage_index != 0 or saved["model"].get("parent_checkpoint"):
                raise ValueError("The first Lightning continuation stage requires an independently fitted parent")
        elif parent.chain_id != self.chain_id or parent.stage_index != self.stage_index - 1:
            raise ValueError("Expected the previous Lightning stage from the same chain")
        if isinstance(reference, dict) and (dataset := reference.get("dataset")):
            inputs = saved.get("provenance", {}).get("inputs", {})
            sources = {item["dataset"] for item in inputs.values() if item.get("split") == "train"}
            if sources != {dataset}:
                raise ValueError(f"Expected a Lightning parent trained on {dataset}")
        if "ridge_lambda" in settings and self.ridge_lambda != parent.ridge_lambda:
            raise ValueError("Continuation ridge_lambda must match the parent")
        self.model = parent.model
        self.ridge_lambda = parent.ridge_lambda
        self.training_ids = set(parent.training_ids)
        self.encoder_spec = {**parent.encoder_spec, **self.encoder_spec}
        self._processor_state = parent._processor_state

    @staticmethod
    def _asset(value):
        spec = value if isinstance(value, dict) else {'path': value}
        path = Path(os.path.expandvars(spec['path'])).expanduser().resolve(strict=True)
        if spec.get("sha256") and sha256(path) != spec["sha256"]:
            raise ValueError("Lightning source asset checksum differs")
        return path

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: LightningTask._numpy(v) for k, v in value.items()}
        return value

    @staticmethod
    def _tensors(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, dict):
            return {k: LightningTask._tensors(v) for k, v in value.items()}
        return value

    def _encoder_options(self):
        options = {key: value for key, value in self.encoder_spec.items() if key != "model_id"}
        if "revision" not in options:
            from huggingface_hub import resolve_revision

            revision = resolve_revision(CLIP_MODEL_ID, **options).resolved
            self.encoder_spec["revision"] = options["revision"] = revision
        return options

    def _ensure_processor(self):
        if self.processor is None:
            from transformers import CLIPImageProcessorPil
            if self._processor_state is not None:
                self.processor = CLIPImageProcessorPil.from_dict(self._processor_state)
            else:
                self.processor = CLIPImageProcessorPil.from_pretrained(CLIP_MODEL_ID, **self._encoder_options())
                self._processor_state = self.processor.to_dict()
        return self.processor

    def _ensure_backbone(self, runtime):
        self._validate_runtime(runtime)
        if self.backbone is None:
            self.backbone = CLIPEncoder(load_clip_vision(CLIP_MODEL_ID, **self._encoder_options()))
        if self._device != runtime.device:
            self.backbone.to(runtime.device).eval(); self._device = runtime.device
            self.detector = None
        return self.backbone

    @staticmethod
    def _validate_runtime(runtime):
        if runtime.precision != 'fp32':
            raise ValueError('Formal Lightning requires strict FP32')
        if runtime.device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32
                                               or torch.backends.cudnn.allow_tf32):
            raise ValueError('Formal Lightning requires TF32 disabled')

    def get_collate_fn(self, stage):
        from data.loader import ImageCollator
        return ImageCollator(self._ensure_processor())

    def _readouts(self, batch, runtime):
        backbone = self._ensure_backbone(runtime)
        features = backbone(batch['pixel_values'].to(runtime.device, non_blocking=True))
        return {b: features[b].float().cpu().numpy() for b in BRANCHES}

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._validate_runtime(runtime)
        if "ridge_lambda" in settings:
            raise ValueError("Declare ridge_lambda under model")
        if loader.dataset.header["split"] != "train":
            raise ValueError("Lightning fitting requires a training input")
        if any(settings.get(key) is not None for key in ("epochs", "patience", "monitor")):
            raise ValueError("Lightning fits analytically without epochs or validation-based selection")
        if self.model is None:
            accumulators = {b: MappingAccumulator() for b in BRANCHES}
            for batch in loader:
                features = self._readouts(batch, runtime)
                labels = batch["labels"].cpu().numpy()
                for branch, accumulator in accumulators.items():
                    accumulator.update(features[branch], labels)
            model = AnalyticReadout({b: accumulator.fit(PROJECTION_RANK)[0] for b, accumulator in accumulators.items()})
            self._validate_mapping(model)
        else:
            if self.stage_index == 0 and self.model.statistics:
                raise ValueError("Source statistics can only be fitted once")
            model = AnalyticReadout(self.model.mappings, self.model.statistics.copy())

        seen = set()
        for batch in loader:
            ids = [item['id'] for item in batch['metadata']]
            if len(set(ids)) != len(ids) or seen.intersection(ids) or self.training_ids.intersection(ids):
                raise ValueError('Duplicate/replayed source IDs would double-count statistics')
            seen.update(ids)
            model.update(self._readouts(batch, runtime), batch['labels'].cpu().numpy())
        if not seen:
            raise ValueError('No successful source images')
        audits = model.solve(self.ridge_lambda)
        self.model = model
        self.training_ids.update(seen)
        self.detector = None
        from engine.evaluate import coverage, evaluate
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        report = {'fit_kind':'analytic_fixed8_equal_raw_cls_mean', 'epochs':None,
                  'new_training_images':len(seen), 'cumulative_training_images':len(self.training_ids),
                  'old_image_replay':0, 'validation_used_for_selection':False,
                  'normal_equation_checks':audits,
                  'coverage':coverage(loader.dataset, seen, output)}
        if validation_loader is not None:
            report['source_validation'] = evaluate(self, validation_loader, runtime=runtime,
                settings={'threshold':0.,'video_aggregation':'mean','group_by':[]},
                output_dir=output/'source-validation')
        return report

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if self.model is None or self.model.coefficients is None:
            raise ValueError('Fit or restore a Lightning checkpoint before prediction')
        backbone = self._ensure_backbone(runtime)
        if self.detector is None:
            self.detector = LightningDetector(backbone, self.model.coefficients).to(runtime.device).eval()
        return self.detector(images.to(runtime.device, non_blocking=True))

    def state_dict(self):
        if self.model is None or self.model.coefficients is None:
            raise ValueError('Cannot save an unfitted fixed-eight head')
        self._ensure_processor()
        return {**_CHECKPOINT_HEADER,'layers':list(LAYERS),
                'encoder':dict(self.encoder_spec),'processor':self._processor_state,
                'mapping':self._tensors({b:m.state_dict() for b,m in self.model.mappings.items()}),
                'statistics':self._tensors({b:s.state_dict() for b,s in self.model.statistics.items()}),
                'coefficients':self._tensors(self.model.coefficients),
                'regularization':self.regularization,
                'training_ids':sorted(self.training_ids)}
