"""Fit, evaluate, and save Lightning's Ridge readout with the shared engine."""
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from backbones.load_backbone import load_backbone, load_image_processor, resolve_backbone_revision
from engine.resources import resolve_asset

from .detector import LightningDetector
from .encoder import DEFAULT_BACKBONE_ID, DEFAULT_LAYERS, VisionEncoder
from .model import (
    DEFAULT_RIDGE_LAMBDA, PROJECTION_RANK,
    AnalyticReadout, MappingAccumulator, SpectralMapping, SufficientStatistics,
    positive_ridge_lambda,
)


def _layer_indices(layers):
    if (not isinstance(layers, (list, tuple)) or not layers
            or any(type(index) is not int or index < 0 for index in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("Lightning layers must be a nonempty list of distinct zero-based block indices")
    return tuple(layers)


class LightningTask:
    """Fit a readout or restore a fitted checkpoint for evaluation.

    A fresh fit selects a backbone and its layers. Set checkpoint to continue
    with the saved encoder, maps and source statistics.
    """

    def __init__(self, settings, *, state=None):
        self.readout = None
        self.training_ids = set()

        # Build the runtime components on first use.
        self.backbone = self.detector = self.processor = None
        self._processor_state = None

        if state is not None:
            self._restore_state(state)
        elif checkpoint := settings.get("checkpoint"):
            self._continue_fit(checkpoint, settings)
        else:
            self._start_fit(settings)

    def _start_fit(self, settings):
        self.backbone_id = str(settings.get("backbone_id", DEFAULT_BACKBONE_ID))
        self.encoder_revision = settings.get("revision")
        self.layers = _layer_indices(settings.get("layers", DEFAULT_LAYERS))
        self.ridge_lambda = positive_ridge_lambda(settings.get("ridge_lambda", DEFAULT_RIDGE_LAMBDA))
        # Without a supplied mapping, fit() learns it from the source data.
        if "mapping_path" in settings:
            self.readout = AnalyticReadout.from_mapping_file(resolve_asset(settings["mapping_path"]))
            self._check_mapping_shape(self.readout)

    def _continue_fit(self, checkpoint, settings):
        from engine.fit import restore

        if "mapping_path" in settings:
            raise ValueError("Lightning continuation uses the mapping stored in the checkpoint")
        previous, saved = restore(checkpoint)
        if not isinstance(previous, type(self)):
            raise ValueError("Lightning continuation requires a fitted Lightning checkpoint")
        if isinstance(checkpoint, dict) and (dataset := checkpoint.get("dataset")):
            inputs = saved.get("provenance", {}).get("inputs", {})
            sources = {item["dataset"] for item in inputs.values() if item.get("split") == "train"}
            if sources != {dataset}:
                raise ValueError(f"Expected a Lightning checkpoint trained on {dataset}")
        if settings.get("ridge_lambda", previous.ridge_lambda) != previous.ridge_lambda:
            raise ValueError("Continuation ridge_lambda must match the checkpoint")
        if str(settings.get("backbone_id", previous.backbone_id)) != previous.backbone_id:
            raise ValueError("Continuation backbone_id must match the checkpoint")
        if _layer_indices(settings.get("layers", previous.layers)) != previous.layers:
            raise ValueError("Continuation layers must match the checkpoint")
        if "revision" in settings:
            revision = resolve_backbone_revision(previous.backbone_id, settings["revision"])
            if revision != previous.encoder_revision:
                raise ValueError("Continuation revision must match the checkpoint")
        self.ridge_lambda = previous.ridge_lambda
        self.readout = previous.readout
        self.training_ids = set(previous.training_ids)
        self.backbone_id = previous.backbone_id
        self.encoder_revision = previous.encoder_revision
        self.layers = previous.layers
        self._processor_state = previous._processor_state

    def _check_mapping_shape(self, readout):
        width = readout.mappings["mean"].mean.shape[1]
        expected = (len(self.layers), width, PROJECTION_RANK)
        if any(mapping.basis.shape != expected for mapping in readout.mappings.values()):
            raise ValueError(f"Both Lightning maps must have basis shape {expected}")
        return expected[:2]

    def _restore_state(self, state):
        self.layers = _layer_indices(state["layers"])
        self.ridge_lambda = positive_ridge_lambda(state.get("ridge_lambda"))
        encoder = state["encoder"]
        self.backbone_id = encoder["model_id"]
        self.encoder_revision = encoder.get("revision")
        self._processor_state = state["processor"]
        ids = state["training_ids"]
        self.training_ids = set(ids)
        mappings = {branch: SpectralMapping(**mapping) for branch, mapping in state["mapping"].items()}
        statistics = {branch: SufficientStatistics.from_state(stats)
                      for branch, stats in state["statistics"].items()}
        coefficients = {name: np.asarray(value) for name, value in state["coefficients"].items()}
        readout = AnalyticReadout(mappings, statistics, coefficients)
        feature_shape = self._check_mapping_shape(readout)
        if set(statistics) != set(readout.branches) or set(coefficients) != {*readout.branches, "bias"}:
            raise ValueError("A fitted Lightning checkpoint must contain both complete readout branches")
        width = len(self.layers) * PROJECTION_RANK
        if (len(ids) != len(self.training_ids)
                or any(s.n != len(ids) or not 0 < s.sy < s.n
                       or s.g.shape != (width, width) or s.h.shape != (width,) or s.sz.shape != (width,)
                       or not all(np.isfinite(a).all() for a in (s.g, s.h, s.sz))
                       for s in statistics.values())):
            raise ValueError("Lightning source statistics do not match its training IDs or projected width")
        if (any(coefficients[branch].shape != feature_shape for branch in readout.branches)
                or np.shape(coefficients["bias"]) != ()
                or not all(np.isfinite(a).all() for a in coefficients.values())):
            raise ValueError("Invalid Lightning coefficients")
        self.readout = readout

    @cached_property
    def _resolved_revision(self):
        self.encoder_revision = resolve_backbone_revision(self.backbone_id, self.encoder_revision)
        return self.encoder_revision

    def _get_processor(self):
        if self.processor is None:
            self.processor = load_image_processor(
                self.backbone_id, revision=self._resolved_revision, state=self._processor_state,
            )
            if self._processor_state is None:
                self._processor_state = self.processor.to_dict()
        return self.processor

    def _get_backbone(self, device):
        if self.backbone is None:
            vision = load_backbone(self.backbone_id, revision=self._resolved_revision)
            backbone = VisionEncoder(vision, self.layers).eval()
            if self.readout is not None:
                width = self.readout.mappings["mean"].mean.shape[1]
                if width != backbone.width or self.readout.branches != (backbone.global_branch, "mean"):
                    raise ValueError("Lightning mapping does not match the backbone's width or readout branches")
            self.backbone = backbone
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
        return {branch: values.float().cpu().numpy() for branch, values in features.items()}

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        if "ridge_lambda" in settings:
            raise ValueError("Declare ridge_lambda under model")
        if loader.dataset.header["split"] != "train":
            raise ValueError("Lightning fitting requires a training input")
        if any(settings.get(key) is not None for key in ("epochs", "patience", "monitor")):
            raise ValueError("Lightning fits analytically without epochs or validation-based selection")
        if self.readout is None:
            accumulators = {}
            for batch in loader:
                features = self._readouts(batch, runtime)
                if not accumulators:
                    accumulators = {branch: MappingAccumulator() for branch in features}
                labels = batch["labels"].cpu().numpy()
                for branch, accumulator in accumulators.items():
                    accumulator.update(features[branch], labels)
            if not accumulators:
                raise ValueError("No successful source images")
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
            "layers": list(self.layers),
            "encoder": {"model_id": self.backbone_id, "revision": self.encoder_revision},
            "processor": self._processor_state,
            "mapping": {branch: {name: torch.tensor(value) for name, value in mapping.state_dict().items()}
                        for branch, mapping in self.readout.mappings.items()},
            "statistics": {branch: {name: torch.tensor(value) for name, value in stats.state_dict().items()}
                           for branch, stats in self.readout.statistics.items()},
            "coefficients": {name: torch.tensor(value) for name, value in self.readout.coefficients.items()},
            "ridge_lambda": self.ridge_lambda,
            "training_ids": sorted(self.training_ids),
        }
