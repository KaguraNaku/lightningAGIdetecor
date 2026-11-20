"""Shared-engine Task for the formal fixed-eight Lightning protocol.

The frozen encoder is a pinned shared resource. Checkpoints contain only the
small readout head, fixed mapping, sufficient statistics and processor identity;
they never duplicate encoder weights or persist per-image feature matrices.
"""
from pathlib import Path
import hashlib
import math
import os

import numpy as np
import torch

from models.clip import load_clip_vision

from .model import (BRANCHES, LAYERS, AnalyticReadout, MappingAccumulator,
                     SpectralMapping, SufficientStatistics)

from .encoder import CLIPEncoder
from .detector import LightningDetector


class LightningTask:
    VERSION = 1
    MODEL_ID = 'openai/clip-vit-large-patch14'
    REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
    prediction_unit = 'image'

    def __init__(self, settings, *, state=None):
        self.settings = dict(settings)
        self.backbone = self.detector = self.processor = None
        self._processor_state = None
        self._device = None
        self.model = None
        self.training_ids = set()
        self.regularization = {"kind": "uniform_mean_loss", "ridge_lambda": settings.get("ridge_lambda", .15)}
        if state is None and any(key in settings for key in ("block_penalties", "eta", "layer_penalties", "regularization")):
            raise ValueError("Lightning supports only uniform ridge_lambda; layered regularization was removed")
        if state is not None:
            if state.get('format') != 'fixed8-cls-mean' or state.get('version') != self.VERSION:
                raise ValueError('Unsupported fixed-eight Lightning checkpoint')
            if tuple(state['layers']) != LAYERS:
                raise ValueError('Saved fixed-eight layers differ from protocol')
            self.encoder_spec = state['encoder']
            self._processor_state = state['processor']
            self.training_ids = set(state.get('training_ids', []))
            self.regularization = state.get('regularization', {"kind": "historical_readonly"})
            mappings = {b: SpectralMapping(**self._numpy(state['mapping'][b])) for b in BRANCHES}
            statistics = {b: SufficientStatistics.from_state(self._numpy(s))
                          for b, s in state.get('statistics', {}).items()}
            self.model = AnalyticReadout(mappings, statistics, self._numpy(state['coefficients']))
        else:
            self.encoder_spec = {
                'model_id': settings.get('backbone_id', self.MODEL_ID),
                'revision': settings.get('revision', self.REVISION),
                'local_files_only': settings.get('local_files_only', True),
            }
            if settings.get('cache_dir'):
                self.encoder_spec['cache_dir'] = str(Path(settings['cache_dir']).expanduser())
            if settings.get('mapping_path'):
                self.model = AnalyticReadout.from_npz(
                    self._asset(settings['mapping_path']),
                    self._asset(settings['coefficients_path']) if settings.get('coefficients_path') else None,
                    {b: self._asset(p) for b, p in settings.get('statistics', {}).items()})
                if settings.get('coefficients_path') or settings.get('statistics'):
                    self.regularization = {"kind": "historical_readonly"}
            if settings.get('parent_checkpoint'):
                from engine.fit import restore
                parent, _ = restore(self._asset(settings['parent_checkpoint']))
                if not isinstance(parent, type(self)) or not parent.model.statistics:
                    raise ValueError('Incremental Lightning requires a fixed-eight parent with statistics')
                self.model = parent.model
                if ("ridge_lambda" in settings
                        and settings["ridge_lambda"] != parent.regularization.get("ridge_lambda")):
                    raise ValueError("Incremental uniform ridge_lambda must match the parent")
                self.regularization = parent.regularization.copy()
                self.training_ids = set(parent.training_ids)
                self.encoder_spec, self._processor_state = parent.encoder_spec, parent._processor_state
        if (self.encoder_spec['model_id'] != self.MODEL_ID
                or self.encoder_spec['revision'] != self.REVISION):
            raise ValueError('The formal fixed-eight encoder identity is pinned')
        if self.regularization.get("kind") == "uniform_mean_loss":
            value = self.regularization.get("ridge_lambda")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Uniform ridge_lambda must be finite and positive")

    @staticmethod
    def _asset(value):
        spec = value if isinstance(value, dict) else {'path': value}
        path = Path(os.path.expandvars(spec['path'])).expanduser().resolve(strict=True)
        if spec.get('sha256'):
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                    raise ValueError('Lightning source asset checksum differs')
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

    def _ensure_processor(self):
        if self.processor is None:
            from transformers import CLIPImageProcessorPil
            if self._processor_state is not None:
                self.processor = CLIPImageProcessorPil.from_dict(self._processor_state)
            else:
                self.processor = CLIPImageProcessorPil.from_pretrained(**{
                    'pretrained_model_name_or_path': self.encoder_spec['model_id'],
                    **{k: v for k, v in self.encoder_spec.items() if k != 'model_id'}})
                self._processor_state = self.processor.to_dict()
        return self.processor

    def _ensure_backbone(self, runtime):
        self._validate_runtime(runtime)
        if self.backbone is None:
            spec = dict(self.encoder_spec)
            model_id = spec.pop('model_id')
            self.backbone = CLIPEncoder(load_clip_vision(model_id, **spec))
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
        if self.regularization.get("kind") != "uniform_mean_loss":
            raise ValueError("Historical Lightning heads are inference-only; start a new uniform-ridge fit")
        if any(key in settings for key in ("eta", "block_penalties", "layer_penalties", "ridge_lambda")):
            raise ValueError("Declare the single uniform coefficient as model.ridge_lambda")
        route = self.settings.get('route', 'ffpp')
        if route not in ('ffpp', 'fresh_UFD', 'incremental_FFpp_UFD'):
            raise ValueError('Unknown fixed-eight training route')
        if loader.dataset.header['split'] != 'train':
            raise ValueError('Only source training labels enter analytic statistics')
        if settings.get('patience') is not None or settings.get('monitor') is not None:
            raise ValueError('Lightning has no early stopping or validation-based head selection')
        if route == 'fresh_UFD':
            if self.model is not None or self.training_ids:
                raise ValueError('Fresh UFD heads cannot inherit source statistics')
            accumulators = {b: MappingAccumulator() for b in BRANCHES}
            for batch in loader:
                features = self._readouts(batch, runtime); labels = batch['labels'].cpu().numpy()
                for b in BRANCHES:accumulators[b].update(features[b], labels)
            self.model = AnalyticReadout({b: accumulators[b].fit(128)[0] for b in BRANCHES})
        elif self.model is None:
            raise ValueError('FF++/incremental routes require their frozen mapping or saved parent')
        if route == 'ffpp' and self.model.statistics:
            raise ValueError('FF++ source statistics can only be added once')
        if route == 'incremental_FFpp_UFD' and not self.model.statistics:
            raise ValueError('Incremental route must retain previous source statistics')
        seen = set()
        for batch in loader:
            ids = [item['id'] for item in batch['metadata']]
            if len(set(ids)) != len(ids) or seen.intersection(ids) or self.training_ids.intersection(ids):
                raise ValueError('Duplicate/replayed source IDs would double-count statistics')
            seen.update(ids)
            self.model.update(self._readouts(batch, runtime), batch['labels'].cpu().numpy())
        if not seen:
            raise ValueError('No successful source images')
        self.training_ids.update(seen)
        audits = self.model.solve(self.regularization["ridge_lambda"]);self.detector = None
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
            raise ValueError('Fit or restore a fixed-eight head before prediction')
        backbone = self._ensure_backbone(runtime)
        if self.detector is None:
            self.detector = LightningDetector(backbone, self.model.coefficients).to(runtime.device).eval()
        return self.detector(images.to(runtime.device, non_blocking=True))

    def state_dict(self):
        if self.model is None or self.model.coefficients is None:
            raise ValueError('Cannot save an unfitted fixed-eight head')
        self._ensure_processor()
        return {'format':'fixed8-cls-mean','version':self.VERSION,'layers':list(LAYERS),
                'encoder':dict(self.encoder_spec),'processor':self._processor_state,
                'mapping':self._tensors({b:m.state_dict() for b,m in self.model.mappings.items()}),
                'statistics':self._tensors({b:s.state_dict() for b,s in self.model.statistics.items()}),
                'coefficients':self._tensors(self.model.coefficients),
                'regularization':dict(self.regularization),
                'training_ids':sorted(self.training_ids)}
