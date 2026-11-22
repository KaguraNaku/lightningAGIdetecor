"""Fit, evaluate, and save Lightning's Ridge readout with the shared engine."""
from pathlib import Path
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
    positive_ridge_lambda,
)


class LightningTask:
    """Fit a readout or restore a fitted checkpoint for evaluation.

    A fresh fit accepts optional ridge_lambda and mapping_path. Set checkpoint
    to continue from a previous fit's map and statistics.
    """

    def __init__(self, settings, *, state=None):
        self.readout = None
        self.training_ids = set()

        # Build the runtime components on first use.
        self.encoder_revision = None
        self.backbone = self.detector = self.processor = None
        self._processor_state = None

        if state is not None:
            self._restore_state(state)
        elif checkpoint := settings.get("checkpoint"):
            self._continue_fit(checkpoint, settings)
        else:
            self._start_fit(settings)

    def _start_fit(self, settings):
        self.ridge_lambda = positive_ridge_lambda(settings.get("ridge_lambda", DEFAULT_RIDGE_LAMBDA))
        # Without a supplied mapping, fit() learns it from the source data.
        if "mapping_path" in settings:
            self.readout = AnalyticReadout.from_mapping_file(self._asset(settings["mapping_path"]))
            self._check_mapping_shape(self.readout)

    def _continue_fit(self, checkpoint, settings):
        if "mapping_path" in settings:
            raise ValueError("Lightning continuation uses the mapping stored in the checkpoint")
        previous = self._load_checkpoint(checkpoint)
        if settings.get("ridge_lambda", previous.ridge_lambda) != previous.ridge_lambda:
            raise ValueError("Continuation ridge_lambda must match the checkpoint")
        self.ridge_lambda = previous.ridge_lambda
        self.readout = previous.readout
        self.training_ids = set(previous.training_ids)
        self.encoder_revision = previous.encoder_revision
        self._processor_state = previous._processor_state

    def _check_mapping_shape(self, readout):
        expected = (len(LAYERS), CLIP_WIDTH, PROJECTION_RANK)
        if any(mapping.basis.shape != expected for mapping in readout.mappings.values()):
            raise ValueError(f"Both Lightning maps must have basis shape {expected}")

    def _restore_state(self, state):
        if tuple(state.get("layers", ())) != LAYERS:
            raise ValueError("Lightning checkpoint uses different CLIP layers")
        self.ridge_lambda = positive_ridge_lambda(state.get("ridge_lambda"))
        encoder = state["encoder"]
        if encoder["model_id"] != CLIP_MODEL_ID:
            raise ValueError("Lightning uses the fixed CLIP-L/14 encoder")
        self.encoder_revision = encoder.get("revision")
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
        self.readout = AnalyticReadout(mappings, statistics, coefficients)
        self._check_mapping_shape(self.readout)

    def _load_checkpoint(self, reference):
        from engine.fit import restore

        previous, saved = restore(self._asset(reference))
        if not isinstance(previous, type(self)):
            raise ValueError("Lightning continuation requires a fitted Lightning checkpoint")
        if isinstance(reference, dict) and (dataset := reference.get("dataset")):
            inputs = saved.get("provenance", {}).get("inputs", {})
            sources = {item["dataset"] for item in inputs.values() if item.get("split") == "train"}
            if sources != {dataset}:
                raise ValueError(f"Expected a Lightning checkpoint trained on {dataset}")
        return previous

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

    def _clip_revision(self):
        if self.encoder_revision is None:
            from huggingface_hub import resolve_revision

            self.encoder_revision = resolve_revision(CLIP_MODEL_ID).resolved
        return self.encoder_revision

    def _get_processor(self):
        if self.processor is None:
            from transformers import CLIPImageProcessorPil
            if self._processor_state is not None:
                self.processor = CLIPImageProcessorPil.from_dict(self._processor_state)
            else:
                self.processor = CLIPImageProcessorPil.from_pretrained(
                    CLIP_MODEL_ID, revision=self._clip_revision()
                )
                self._processor_state = self.processor.to_dict()
        return self.processor

    def _get_backbone(self, device):
        if self.backbone is None:
            self.backbone = CLIPEncoder(
                load_clip_vision(CLIP_MODEL_ID, revision=self._clip_revision())
            ).eval()
        if next(self.backbone.parameters()).device != device:
            self.backbone.to(device)
        return self.backbone

    def get_collate_fn(self, stage):
        from data.loader import ImageCollator
        return ImageCollator(self._get_processor())

    def _readouts(self, batch, runtime):
        images = batch["pixel_values"].to(runtime.device, non_blocking=True)
        backbone = self._get_backbone(images.device)
        with runtime.autocast():
            features = backbone(images)
        return {b: features[b].float().cpu().numpy() for b in BRANCHES}

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        if "ridge_lambda" in settings:
            raise ValueError("Declare ridge_lambda under model")
        if loader.dataset.header["split"] != "train":
            raise ValueError("Lightning fitting requires a training input")
        if any(settings.get(key) is not None for key in ("epochs", "patience", "monitor")):
            raise ValueError("Lightning fits analytically without epochs or validation-based selection")
        if self.readout is None:
            accumulators = {b: MappingAccumulator() for b in BRANCHES}
            for batch in loader:
                features = self._readouts(batch, runtime)
                labels = batch["labels"].cpu().numpy()
                for branch, accumulator in accumulators.items():
                    accumulator.update(features[branch], labels)
            mappings = {b: accumulator.fit(PROJECTION_RANK)[0] for b, accumulator in accumulators.items()}
            readout = AnalyticReadout(mappings)
        else:
            readout = AnalyticReadout(self.readout.mappings, self.readout.statistics.copy())

        seen = set()
        for batch in loader:
            ids = [item['id'] for item in batch['metadata']]
            if len(set(ids)) != len(ids) or seen.intersection(ids) or self.training_ids.intersection(ids):
                raise ValueError('Duplicate/replayed source IDs would double-count statistics')
            seen.update(ids)
            readout.update(self._readouts(batch, runtime), batch['labels'].cpu().numpy())
        if not seen:
            raise ValueError('No successful source images')
        audits = readout.solve(self.ridge_lambda)
        self.readout = readout
        self.training_ids.update(seen)
        self.detector = None
        from engine.evaluate import coverage, evaluate
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        report = {'fit_kind':'linear_solve', 'epochs':None,
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
        if self.readout is None or self.readout.coefficients is None:
            raise ValueError('Fit or restore a Lightning checkpoint before prediction')
        images = images.to(runtime.device, non_blocking=True)
        backbone = self._get_backbone(images.device)
        if self.detector is None:
            self.detector = LightningDetector(backbone, self.readout.coefficients).eval()
        if self.detector.bias.device != images.device:
            self.detector.to(images.device)
        with runtime.autocast():
            return self.detector(images)

    def state_dict(self):
        if self.readout is None or self.readout.coefficients is None:
            raise ValueError("Fit Lightning before saving a checkpoint")
        if self._processor_state is None:
            self._get_processor()
        return {
            "layers": list(LAYERS),
            "encoder": {"model_id": CLIP_MODEL_ID, "revision": self.encoder_revision},
            "processor": self._processor_state,
            "mapping": self._tensors({b: m.state_dict() for b, m in self.readout.mappings.items()}),
            "statistics": self._tensors({b: s.state_dict() for b, s in self.readout.statistics.items()}),
            "coefficients": self._tensors(self.readout.coefficients),
            "ridge_lambda": self.ridge_lambda,
            "training_ids": sorted(self.training_ids),
        }
