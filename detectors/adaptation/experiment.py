"""Compose native detector tasks with explicit ProGAN source supervision."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler

from engine.runtime import create_task

ALLOWED = {'xception', 'f3net', 'core', 'ucf', 'prodet', 'effort', 'gend', 'forada'}
DOMAINS = {'real': 0, 'FF_Deepfakes': 1, 'FF_Face2Face': 2,
           'FF_FaceSwap': 3, 'FF_NeuralTextures': 4, 'ProGAN': 5}


class WholeImageForAdaDataset:
    """Fully generated images have full-image masks; genuine images have zero masks.

    This explicit UFD extension is not a claim of annotated manipulation boundaries.
    The native collator/loss, including its boundary mathematics, remains unchanged.
    """
    def __init__(self, base):
        if base.header['dataset'] != 'UniversalFakeDetect' or base.header['split'] != 'train':
            raise ValueError('Whole-image supervision is restricted to declared UFD training')
        self.base, self.records, self.header = base, base.records, base.header
        self.root, self.manifest_path = base.root, base.manifest_path
        if any(r.label == 1 and r.metadata.get('generator') != 'ProGAN' for r in self.records):
            raise ValueError('Fake UFD rows require explicit generator=ProGAN metadata')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if sample is None:
            return None
        mask = np.full((256, 256), 255. if self.records[index].label else 0., dtype=np.float32)
        return {**sample, 'forada_mask': mask}

    def selection_receipt(self):
        return {'mask_policy': 'UFD explicit whole-image synthetic255/genuine0;derived from image labels',
                'boundary_claim': 'No localized manipulation boundary annotation exists;native boundary targets may degenerate',
                'source_manifest': str(self.manifest_path)}


class FakeGuardSampler(Sampler):
    """Shuffle without replacement and reserve at least one fake per native batch."""
    def __init__(self, records, batch_size, seed):
        self.real = [i for i, r in enumerate(records) if r.label == 0]
        self.fake = [i for i, r in enumerate(records) if r.label == 1]
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.size = len(records)
        if len(self.fake) < (self.size + batch_size - 1) // batch_size:
            raise ValueError('Not enough fake images to guard every ForAda batch')

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        real = [self.real[i] for i in torch.randperm(len(self.real), generator=rng).tolist()]
        fake = [self.fake[i] for i in torch.randperm(len(self.fake), generator=rng).tolist()]
        ri = fi = 0
        while ri + fi < self.size:
            left = self.size - ri - fi
            size = min(self.batch_size, left)
            steps = (left + self.batch_size - 1) // self.batch_size
            nf, nr = len(fake) - fi, len(real) - ri
            take_fake = max(1, round(size * nf / left), size - nr)
            take_fake = min(take_fake, size, nf - (steps - 1))
            take_real = size - take_fake
            if not (0 <= take_real <= nr and 1 <= take_fake <= nf):
                raise ValueError('Cannot construct native ForAda batch without replacement')
            batch = real[ri:ri + take_real] + fake[fi:fi + take_fake]
            for i in torch.randperm(len(batch), generator=rng).tolist():
                yield batch[i]
            ri += take_real
            fi += take_fake


def expand_ucf(task):
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
    task.construction['specific_label_map'] = dict(DOMAINS)


class UFDAdaptationTask:
    def __init__(self, settings, *, state=None):
        if state is not None:
            if state.get('version') != 1:
                raise ValueError('Unsupported UFD adaptation state')
            self.base_model = deepcopy(state['base_model'])
            self.adaptation = deepcopy(state['adaptation'])
            self.task = create_task(self.base_model, state=state['base_task'])
        else:
            self.base_model = deepcopy(settings['base_model'])
            method = self.base_model['factory'].split('.')[1]
            if method not in ALLOWED:
                raise ValueError('Only the eight authorized detectors are supported')
            self.adaptation = {'regime': settings['regime'], 'source': 'UFD/ProGAN',
                'optimizer': 'Fresh native optimizer per fit;not exact optimizer/RNG resume',
                'UCF_domain_map': dict(DOMAINS),
                'ForAda_supervision': 'whole-image mask for fully generated images;native losses retained'}
            parent = settings.get('parent_checkpoint')
            if parent:
                path = Path(parent)
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != settings['parent_sha256']:
                    raise ValueError('FF++ parent checkpoint checksum differs')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                if checkpoint['model']['factory'] != self.base_model['factory']:
                    raise ValueError('Parent method differs')
                self.task = create_task(checkpoint['model'], state=checkpoint['task'])
                self.adaptation['parent'] = {'path': str(path), 'sha256': actual,
                    'selected_epoch': checkpoint['fit_report']['best_epoch']}
                if method == 'ucf':
                    expand_ucf(self.task)
                    self.base_model.update(specific_classes=6, specific_label_map=dict(DOMAINS))
            else:
                if method == 'ucf':
                    self.base_model.update(specific_classes=6, specific_label_map=dict(DOMAINS))
                self.task = create_task(self.base_model)
            self.task.origin['UFD_adaptation'] = deepcopy(self.adaptation)
        self.method = self.base_model['factory'].split('.')[1]

    def __getattr__(self, name):
        if name == 'task':
            raise AttributeError(name)
        return getattr(self.task, name)

    def get_collate_fn(self, stage):
        return self.task.get_collate_fn(stage)

    def prepare_dataset(self, dataset, *, stage):
        if stage == 'train' and self.method == 'forada':
            return WholeImageForAdaDataset(dataset)
        provider = getattr(self.task, 'prepare_dataset', None)
        return provider(dataset, stage=stage) if provider else dataset

    def get_sampler(self, dataset, *, batch_size, shuffle, seed, stage):
        if stage == 'train' and self.method == 'forada':
            return FakeGuardSampler(dataset.records, batch_size, seed)
        provider = getattr(self.task, 'get_sampler', None)
        return provider(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed, stage=stage) if provider else None

    def fit(self, *args, **kwargs):
        report = self.task.fit(*args, **kwargs)
        return {**report, 'adaptation': deepcopy(self.adaptation)}

    def predict(self, *args, **kwargs):
        return self.task.predict(*args, **kwargs)

    def state_dict(self):
        return {'version': 1, 'base_model': deepcopy(self.base_model),
                'base_task': self.task.state_dict(), 'adaptation': deepcopy(self.adaptation)}


def create_task(settings, *, state=None):
    return UFDAdaptationTask(settings, state=state)
