"""Fit, evaluate, and save Lightning's Ridge readout with the shared engine."""

from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from backbones.load_backbone import (
    get_backbone_model_type,
    load_backbone,
    load_image_processor,
    resolve_backbone_revision,
)
from engine.resources import resolve_asset

from . import config
from .detector import LightningDetector
from .encoder import VisionEncoder, global_readout_for
from .model import (
    AnalyticReadout,
    MappingAccumulator,
    SpectralMapping,
    SufficientStatistics,
    positive_ridge_lambda,
)


def _layer_indices(layers):
    if (
        not isinstance(layers, (list, tuple))
        or not layers
        or any(type(index) is not int or index < 0 for index in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(
            "Lightning layers must be a nonempty list of distinct zero-based block indices"
        )
    return tuple(layers)


class LightningTask:
    """Fit a readout or restore a fitted checkpoint for evaluation.

    A fresh fit uses config.py. Set checkpoint to continue with the saved
    encoder, maps and source statistics.
    """

    default_allow_tf32 = True
    supports_distributed_fit = True
    model_file_fields = ("mapping_path",)

    @staticmethod
    def validate_fit_config(model, fit, *, public):
        if "ridge_lambda" in fit:
            raise ValueError("Set RIDGE_LAMBDA in detectors/lightning/config.py")
        if any(fit.get(key) is not None for key in ("epochs", "patience", "monitor")):
            raise ValueError(
                "Lightning fits analytically without epochs or validation-based selection"
            )

    @staticmethod
    def resume_model_settings(model, path):
        reference = {"path": str(path)}
        configured = model.get("checkpoint")
        if isinstance(configured, dict) and "dataset" in configured:
            reference["dataset"] = configured["dataset"]
        return {"factory": model["factory"], "checkpoint": reference}

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
        if overrides := sorted(settings.keys() - {"mapping_path", "checkpoint"}):
            raise ValueError(
                "Set Lightning model parameters in detectors/lightning/config.py. "
                f"Remove these model fields: {', '.join(overrides)}"
            )
        self.backbone_id = str(config.BACKBONE_ID)
        self.encoder_revision = config.BACKBONE_REVISION
        self.layers = _layer_indices(config.LAYERS)
        self.ridge_lambda = positive_ridge_lambda(config.RIDGE_LAMBDA)
        # Without a supplied mapping, fit() learns it from the source data.
        if "mapping_path" in settings:
            self.readout = AnalyticReadout.from_mapping_file(
                resolve_asset(settings["mapping_path"])
            )
            self._check_mapping_shape(self.readout, rank=config.PROJECTION_RANK)

    def _continue_fit(self, checkpoint, settings):
        from engine.fit import restore

        if overrides := sorted(settings.keys() - {"checkpoint"}):
            raise ValueError(
                f"Lightning continuation only accepts model.checkpoint. Remove: {', '.join(overrides)}"
            )
        previous, saved = restore(checkpoint)
        if not isinstance(previous, type(self)):
            raise ValueError(
                "Lightning continuation requires a fitted Lightning checkpoint"
            )
        if isinstance(checkpoint, dict) and (dataset := checkpoint.get("dataset")):
            inputs = saved.get("provenance", {}).get("inputs", {})
            sources = {
                item["dataset"]
                for item in inputs.values()
                if item.get("split") == "train"
            }
            if sources != {dataset}:
                raise ValueError(
                    f"Expected a Lightning checkpoint trained on {dataset}"
                )
        self.ridge_lambda = previous.ridge_lambda
        self.readout = previous.readout
        self.training_ids = set(previous.training_ids)
        self.backbone_id = previous.backbone_id
        self.encoder_revision = previous.encoder_revision
        self.layers = previous.layers
        self._processor_state = previous._processor_state

    def _check_mapping_shape(self, readout, *, rank):
        width = readout.mappings["mean"].mean.shape[1]
        expected = (len(self.layers), width, rank)
        if any(
            mapping.basis.shape != expected for mapping in readout.mappings.values()
        ):
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
        mappings = {
            branch: SpectralMapping(**mapping)
            for branch, mapping in state["mapping"].items()
        }
        statistics = {
            branch: SufficientStatistics.from_state(stats)
            for branch, stats in state["statistics"].items()
        }
        coefficients = {
            name: np.asarray(value) for name, value in state["coefficients"].items()
        }
        readout = AnalyticReadout(mappings, statistics, coefficients)
        rank = mappings["mean"].rank
        feature_shape = self._check_mapping_shape(readout, rank=rank)
        if set(statistics) != set(readout.branches) or set(coefficients) != {
            *readout.branches,
            "bias",
        }:
            raise ValueError(
                "A fitted Lightning checkpoint must contain both complete readout branches"
            )
        width = len(self.layers) * rank
        if len(ids) != len(self.training_ids) or any(
            stats.sample_count != len(ids)
            or not 0 < stats.label_sum < stats.sample_count
            or stats.gram.shape != (width, width)
            or stats.feature_label_sum.shape != (width,)
            or stats.feature_sum.shape != (width,)
            for stats in statistics.values()
        ):
            raise ValueError(
                "Lightning source statistics do not match its training IDs or projected width"
            )
        if (
            any(
                coefficients[branch].shape != feature_shape
                for branch in readout.branches
            )
            or np.shape(coefficients["bias"]) != ()
            or not all(np.isfinite(a).all() for a in coefficients.values())
        ):
            raise ValueError("Invalid Lightning coefficients")
        self.readout = readout

    @cached_property
    def _backbone_selection(self):
        revision = resolve_backbone_revision(self.backbone_id, self.encoder_revision)
        model_type = get_backbone_model_type(self.backbone_id, revision=revision)
        global_readout_for(model_type)
        self.encoder_revision = revision
        return model_type, revision

    def _get_processor(self):
        if self.processor is None:
            _, revision = self._backbone_selection
            self.processor = load_image_processor(
                self.backbone_id,
                revision=revision,
                state=self._processor_state,
            )
            if self._processor_state is None:
                self._processor_state = self.processor.to_dict()
        return self.processor

    def _get_backbone(self, device):
        if self.backbone is None:
            model_type, revision = self._backbone_selection
            vision = load_backbone(
                self.backbone_id, model_type=model_type, revision=revision
            )
            backbone = VisionEncoder(vision, self.layers).eval()
            if self.readout is not None:
                width = self.readout.mappings["mean"].mean.shape[1]
                if width != backbone.width or self.readout.branches != (
                    backbone.global_branch,
                    "mean",
                ):
                    raise ValueError(
                        "Lightning mapping does not match the backbone's width or readout branches"
                    )
            self.backbone = backbone
        if next(self.backbone.parameters()).device != device:
            self.backbone.to(device)
        return self.backbone

    def get_collate_fn(self, stage):
        from data.loader import ImageCollator

        return ImageCollator(self._get_processor())

    def prepare_distributed(self, group):
        selection = group.broadcast(self._backbone_selection if group.is_main else None)
        self._backbone_selection = selection
        self.encoder_revision = selection[1]

    def _readouts(self, batch, runtime):
        images = batch["pixel_values"].to(runtime.device, non_blocking=True)
        backbone = self._get_backbone(images.device)
        with runtime.autocast():
            features = backbone(images)
        return {
            branch: values.float().cpu().numpy() for branch, values in features.items()
        }

    def _fit_mappings(self, loader, runtime):
        group = runtime.group
        accumulators = {}
        if group is not None:
            backbone = self._get_backbone(runtime.device)
            shape = (len(self.layers), backbone.width)
            accumulators = {
                branch: MappingAccumulator(shape)
                for branch in (backbone.global_branch, "mean")
            }
        for batch in loader:
            with runtime.time_region("fit/mapping/batch", samples=len(batch["labels"])):
                features = self._readouts(batch, runtime)
            if not accumulators:
                accumulators = {branch: MappingAccumulator() for branch in features}
            labels = batch["labels"].cpu().numpy().astype(bool, copy=False)
            for branch, accumulator in accumulators.items():
                accumulator.update(features[branch], labels)
        if not accumulators:
            raise ValueError("No successful source images")

        if group is not None:
            for accumulator in accumulators.values():
                counts = np.array([accumulator.n, accumulator.n_real], dtype=np.int64)
                for array in (
                    counts,
                    accumulator.sum,
                    accumulator.real_sum,
                    accumulator.real_square,
                    accumulator.gram,
                ):
                    group.sum_array(array)
                accumulator.n, accumulator.n_real = counts.tolist()
        mappings = None
        if group is None or group.is_main:
            mappings = {
                branch: accumulator.fit_mapping(config.PROJECTION_RANK)
                for branch, accumulator in accumulators.items()
            }
        return mappings if group is None else group.broadcast(mappings)

    def _merge_statistics(self, readout, prior, group):
        for branch, mapping in readout.mappings.items():
            stats = readout.statistics.get(branch)
            if stats is None:
                stats = SufficientStatistics.empty(len(self.layers) * mapping.rank)
            state = stats.state_dict()
            for value in state.values():
                group.sum_array(value)
            if group.is_main:
                stats = SufficientStatistics.from_state(state)
                readout.statistics[branch] = (
                    prior[branch].add(stats) if branch in prior else stats
                )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        group = runtime.group
        primary = group is None or group.is_main
        mappings = (
            self._fit_mappings(loader, runtime)
            if self.readout is None
            else self.readout.mappings
        )
        prior = {} if self.readout is None else self.readout.statistics
        readout = AnalyticReadout(mappings, prior.copy() if group is None else None)

        seen = set()
        for batch in loader:
            ids = [item["id"] for item in batch["metadata"]]
            if (
                len(set(ids)) != len(ids)
                or seen.intersection(ids)
                or self.training_ids.intersection(ids)
            ):
                raise ValueError(
                    "Duplicate/replayed source IDs would double-count statistics"
                )
            seen.update(ids)
            labels = batch["labels"].cpu().numpy().astype(bool, copy=False)
            with runtime.time_region("fit/statistics/batch", samples=len(labels)):
                features = self._readouts(batch, runtime)
            readout.update(features, labels)
        if group is not None:
            self._merge_statistics(readout, prior, group)
            shard_ids = group.gather(seen)
            if primary:
                seen = set()
                for ids in shard_ids:
                    if seen.intersection(ids):
                        raise ValueError(
                            "Distributed shards contained duplicate source IDs"
                        )
                    seen.update(ids)
        if primary:
            if not seen:
                raise ValueError("No successful source images")
            with runtime.time_region("fit/ridge"):
                audits = readout.solve(self.ridge_lambda)
            self.training_ids.update(seen)
        if group is not None:
            readout.coefficients = group.broadcast(readout.coefficients)
        self.readout = readout
        self.detector = None
        from engine.evaluate import coverage, evaluate

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = None
        if primary:
            report = {
                "fit_kind": "linear_solve",
                "epochs": None,
                "new_training_images": len(seen),
                "cumulative_training_images": len(self.training_ids),
                "old_image_replay": 0,
                "validation_used_for_selection": False,
                "normal_equation_checks": audits,
                "coverage": coverage(loader.dataset, seen, output),
            }
        if validation_loader is not None:
            validation = evaluate(
                self,
                validation_loader,
                runtime=runtime,
                settings={
                    "threshold": 0.0,
                    "video_aggregation": "mean",
                    "group_by": [],
                },
                output_dir=output / "source-validation",
                phase="fit/validation",
            )
            if primary:
                report["source_validation"] = validation
        return report

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if self.readout is None or self.readout.coefficients is None:
            raise ValueError("Fit or restore a Lightning checkpoint before prediction")
        images = images.to(runtime.device, non_blocking=True)
        backbone = self._get_backbone(images.device)
        if self.detector is None:
            self.detector = LightningDetector(
                backbone, self.readout.coefficients
            ).eval()
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
            "encoder": {
                "model_id": self.backbone_id,
                "revision": self.encoder_revision,
            },
            "processor": self._processor_state,
            "mapping": {
                branch: {
                    name: torch.tensor(value)
                    for name, value in mapping.state_dict().items()
                }
                for branch, mapping in self.readout.mappings.items()
            },
            "statistics": {
                branch: {
                    name: torch.tensor(value)
                    for name, value in stats.state_dict().items()
                }
                for branch, stats in self.readout.statistics.items()
            },
            "coefficients": {
                name: torch.tensor(value)
                for name, value in self.readout.coefficients.items()
            },
            "ridge_lambda": self.ridge_lambda,
            "training_ids": sorted(self.training_ids),
        }
