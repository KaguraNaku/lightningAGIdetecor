"""Matched source-selected native training state across disjoint UFD blocks.

Only stage boundaries are supported. Worker processes restart at epoch boundaries
with the carried loader generator; persistent workers are deliberately rejected.
"""
from pathlib import Path
import json
import random

import numpy as np
import torch

from data.manifest import sha256


def qualified(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def records(loader):
    return getattr(loader.dataset, 'all_records', loader.dataset.records)


def loader_state(loader):
    sampler = loader.sampler
    state = {'type': qualified(sampler), 'generator': loader.generator.get_state(),
             'num_workers': loader.num_workers, 'batch_size': loader.batch_size}
    provider = getattr(sampler, 'state_dict', None)
    if callable(provider):
        state['sampler'] = provider()
    generator = getattr(sampler, 'generator', None)
    if isinstance(generator, torch.Generator):
        state['sampler_generator'] = generator.get_state()
    return state


def restore_loader(loader, state):
    if qualified(loader.sampler) != state['type']:
        raise ValueError('Continuation sampler type changed')
    if loader.num_workers != state['num_workers'] or loader.batch_size != state['batch_size']:
        raise ValueError('Continuation native loader protocol changed')
    loader.generator.set_state(state['generator'])
    if 'sampler' in state:
        loader.sampler.load_state_dict(state['sampler'])
    if 'sampler_generator' in state:
        loader.sampler.generator.set_state(state['sampler_generator'])


def optimizer_names(trainer, optimizer):
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


class StageContinuation:
    def __init__(self, settings, loader, validation_loader, runtime):
        self.spec = settings.get('continuation')
        self.enabled = self.spec is not None
        self.steps = self.parent_steps = self.parent_epochs = 0
        self.exposures = self.parent_exposures = 0
        self.actual_ids = set()
        self.parent = None
        if not self.enabled:
            return
        self.loader, self.validation_loader, self.runtime = loader, validation_loader, runtime
        for data in [loader, validation_loader]:
            if data is None or data.persistent_workers or data.drop_last:
                raise ValueError('Stage continuation requires complete epoch-boundary loaders without persistent workers')
            if data.generator is None:
                raise ValueError('Stage continuation needs explicit loader generators')
        self.ids = {r.id for r in records(loader)}
        if len(self.ids) != len(records(loader)):
            raise ValueError('Repeated IDs in a new training block')
        self.validation_sha = sha256(validation_loader.dataset.manifest_path)
        self.stage_index = self.spec['stage_index']
        if type(self.stage_index) is not int or self.stage_index < 0:
            raise ValueError('Invalid continuous stage index')
        if len(self.ids) != self.spec['new_images']:
            raise ValueError('New block size differs from the authorized source budget')
        self.seen_ids = set(self.ids)
        previous = self.spec.get('previous_best')
        if self.stage_index == 0:
            if previous:
                raise ValueError('First UFD stage must create its optimizer once;not restore FF++ optimizer')
        else:
            if not previous or Path(previous['path']).name != 'best.pth':
                raise ValueError('Later stages require the selected best.pth training state')
            if sha256(Path(previous['path'])) != previous['sha256']:
                raise ValueError('Previous selected training checkpoint SHA changed')
            self.parent = torch.load(previous['path'], weights_only=True, mmap=True, map_location='cpu')
            p = self.parent.get('continuation')
            if not p or p['chain_id'] != self.spec['chain_id'] or p['stage_index'] != self.stage_index - 1:
                raise ValueError('Cannot skip, reorder or mix continuous stages')
            if (not self.parent['record']['improved'] or self.parent['epoch'] != previous['selected_epoch']):
                raise ValueError('Best weights must carry their matching selected-epoch optimizer')
            if p['validation_sha256'] != self.validation_sha:
                raise ValueError('Source validation changed across stages')
            if self.ids.intersection(p['seen_ids']):
                raise ValueError('Old training images were replayed in a new block')
            self.seen_ids.update(p['seen_ids'])
            self.steps = self.parent_steps = p['optimizer_steps']
            self.parent_epochs = p['cumulative_epochs']
            self.exposures = self.parent_exposures = p['image_exposures']
            self.actual_ids.update(p['actual_seen_ids'])
        if len(self.seen_ids) != self.spec['cumulative_images']:
            raise ValueError('Disjoint new blocks do not close the cumulative source budget')

    def restore(self, trainer, optimizer, scaler):
        if not self.enabled:
            return
        self.names = optimizer_names(trainer, optimizer)
        self.optimizer_type = qualified(optimizer)
        if self.parent is not None:
            selected, p = self.parent, self.parent['continuation']
            if p['optimizer_type'] != self.optimizer_type or p['optimizer_parameter_names'] != self.names:
                raise ValueError('Native optimizer parameter ordering or type changed')
            before = [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.state_dict()['param_groups']]
            carried = [{k: v for k, v in g.items() if k != 'params'} for g in selected['optimizer']['param_groups']]
            if before != carried:
                raise ValueError('Native optimizer hyperparameters changed at stage boundary')
            trainer.load_selected_weights(selected['state_dict']) if hasattr(trainer, 'load_selected_weights') else trainer.model.load_state_dict(selected['state_dict'], strict=True)
            objective = getattr(trainer, 'objective', None)
            if objective is not None:
                objective.load_state_dict(selected['objective'], strict=True)
            method = getattr(trainer, 'load_method_state_dict', None)
            if callable(method):
                method(selected.get('method_state'))
            optimizer.load_state_dict(selected['optimizer'])
            if not optimizer.state or self.steps <= 0:
                raise ValueError('Refuse a fresh optimizer in a later stage')
            scaler.load_state_dict(selected['scaler'])
            restore_loader(self.loader, p['loaders']['train'])
            restore_loader(self.validation_loader, p['loaders']['validation'])
            torch.set_rng_state(selected['torch_rng'])
            random.setstate(selected['python_rng'])
            n = selected['numpy_rng']
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
            if self.runtime.device.type == 'cuda':
                if len(selected['cuda_rng']) != torch.cuda.device_count():
                    raise ValueError('CUDA RNG device count changed')
                torch.cuda.set_rng_state_all(selected['cuda_rng'])
        optimizer.register_step_post_hook(self._step)
        self.parent = None  # release mmap references once state is loaded

    def _step(self, optimizer, args, kwargs):
        self.steps += 1

    def snapshot(self, snapshot, epoch, training_directory):
        if not self.enabled:
            return
        snapshot.setdefault('objective', {})
        snapshot.setdefault('method_state', None)
        for line in (Path(training_directory) / 'samples.jsonl').read_text().splitlines():
            row = json.loads(line)
            rows = [row['real'], row['fake']] if 'pair' in row else [row]
            for sample in rows:
                if sample['id'] not in self.ids:
                    raise ValueError('Observed replay or an undeclared source image')
                self.actual_ids.add(sample['id'])
                self.exposures += 1
        snapshot['continuation'] = {
            'version': 1, 'chain_id': self.spec['chain_id'], 'stage_index': self.stage_index,
            'new_images': len(self.ids), 'cumulative_images': len(self.seen_ids),
            'seen_ids': sorted(self.seen_ids), 'actual_seen_ids': sorted(self.actual_ids),
            'image_exposures': self.exposures, 'optimizer_steps': self.steps,
            'optimizer_steps_before_stage': self.parent_steps,
            'cumulative_epochs': self.parent_epochs + epoch,
            'validation_sha256': self.validation_sha,
            'new_manifest_sha256': sha256(self.loader.dataset.manifest_path),
            'optimizer_type': self.optimizer_type, 'optimizer_parameter_names': self.names,
            'loaders': {'train': loader_state(self.loader), 'validation': loader_state(self.validation_loader)},
            'old_image_replay': 0, 'worker_state_policy': 'epoch-boundary restart with carried loader RNG',
        }

    def summary(self, selected):
        if not self.enabled:
            return None
        p = selected['continuation']
        return {k: p[k] for k in ['chain_id', 'stage_index', 'new_images', 'cumulative_images',
            'image_exposures', 'optimizer_steps', 'optimizer_steps_before_stage', 'cumulative_epochs', 'old_image_replay']} | {
            'selected_training_checkpoint': 'best.pth', 'selected_epoch': selected['epoch'],
            'attempted_optimizer_steps_this_stage': self.steps - self.parent_steps,
            'attempted_image_exposures_this_stage': self.exposures - self.parent_exposures,
            'selected_actual_unique_images': len(p['actual_seen_ids']),
            'selected_train_sampler': {k: v for k, v in p['loaders']['train'].items() if k not in ['generator', 'sampler_generator']}}
