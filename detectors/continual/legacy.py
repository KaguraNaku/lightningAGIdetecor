"""Explicit import of an already-successful, identical FF++→UFD400 fit.

Old snapshots carried train-loader RNG but not validation-loader RNG or ForAda's
sampler epoch. The latter are reconstructed from the proven epoch-boundary
protocol and recorded; original snapshots and negative later epochs stay intact.
"""
from copy import copy, deepcopy
import json
from pathlib import Path

import torch

from data.manifest import sha256
from detectors._shared.continuity import StageContinuation
from engine.runtime import atomic_save


def validation_rng(seed, completed_iterations):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(completed_iterations):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
    return generator.get_state()


def import_selected(task, loader, validation_loader, *, runtime, old_fit, output, settings):
    old_fit, output = Path(old_fit), Path(output)
    source = torch.load(old_fit / 'best.pth', weights_only=True, mmap=True, map_location='cpu')
    exported = torch.load(old_fit / 'model.pth', weights_only=True, mmap=True, map_location='cpu')
    epoch = exported['fit_report']['best_epoch']
    if source['epoch'] != epoch or not source['record']['improved']:
        raise ValueError('Legacy source-best model and optimizer epoch differ')
    weights = exported['task']['base_task']['state_dict']
    if weights.keys() != source['state_dict'].keys() or any(
            not torch.equal(weights[k], source['state_dict'][k]) for k in weights):
        raise ValueError('Legacy model weights do not match their complete selected training state')
    if settings['continuation']['stage_index'] != 0:
        raise ValueError('Only the first, identical UFD400 fit may be imported')
    if loader.persistent_workers or validation_loader.persistent_workers:
        raise ValueError('Legacy worker state cannot be reconstructed for persistent workers')
    if not torch.__version__.startswith('2.14.1'):
        raise ValueError('Legacy validation RNG reconstruction is verified for pinned torch2.14.1 only')
    loader.generator.set_state(source['loader_rng'])
    seed = source['protocol']['loader_initial_seed']
    validation_loader.generator.set_state(validation_rng(seed, epoch))
    sampler = getattr(loader.sampler, 'load_state_dict', None)
    if callable(sampler):
        state = loader.sampler.state_dict()
        state['epoch'] = epoch
        sampler(state)
    trainable = [p for p in task.model.parameters() if p.requires_grad]
    groups = deepcopy(source['optimizer']['param_groups'])
    if [i for g in groups for i in g['params']] != list(range(len(trainable))):
        raise ValueError('Legacy native optimizer parameter ordering is not explicit/compatible')
    for g in groups:
        g['params'] = [trainable[i] for i in g['params']]
    kind = settings.get('optimizer', 'adam')
    constructor = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}.get(kind)
    if constructor is None:
        raise ValueError('Unrecognized native legacy optimizer')
    optimizer = constructor(groups)
    helper = StageContinuation(settings, loader, validation_loader, runtime)
    # Restore is stage0 initialization of bookkeeping only, not a training update.
    helper.restore(task, optimizer, torch.amp.GradScaler('cuda', enabled=False))
    imported = copy(source)
    history = json.loads((old_fit / 'history.json').read_text())
    for index in range(1, epoch + 1):
        train = old_fit / f'epoch-{index:03d}/train'
        rows = [json.loads(line) for line in (train / 'samples.jsonl').read_text().splitlines()]
        if history[index - 1]['train']['skipped'] != 0:
            raise ValueError('Legacy source coverage is incomplete;cannot certify identical initialization')
        batches = {r['batch'] for r in rows if 'pair' in r}
        helper.steps += len(batches) if batches else (len(rows) + loader.batch_size - 1) // loader.batch_size
        helper.snapshot(imported, index, train)
    if kind == 'adam':
        maximum = max(int(s['step']) for s in source['optimizer']['state'].values())
        if maximum != helper.steps:
            raise ValueError('Legacy optimizer step counters disagree with actual selected training receipts')
    output.mkdir(parents=True, exist_ok=False)
    for path in old_fit.glob('epoch-*'):
        (output / path.name).symlink_to(path, target_is_directory=True)
    for name in ['history.json', 'protocol.json', 'environment.json']:
        (output / name).symlink_to(old_fit / name)
    (output / 'legacy-last.pth').symlink_to(old_fit / 'last.pth')
    imported['legacy_import'] = {
        'original_selected_state': {'path': str(old_fit / 'best.pth'), 'sha256': sha256(old_fit / 'best.pth')},
        'original_selected_model': {'path': str(old_fit / 'model.pth'), 'sha256': sha256(old_fit / 'model.pth')},
        'training_repeated': False, 'selected_epoch': epoch,
        'validation_loader_rng': 'Reconstructed one int64 base_seed draw per completed nonpersistent sequential validation iteration;torch2.14.1 contract-tested',
        'ForAda_sampler_epoch': 'One sampler iteration per completed native epoch;selected epoch count,not stopping epoch',
    }
    atomic_save(output / 'best.pth', imported, torch_format=True)
    exported = copy(exported)
    exported['model'] = settings['import_model_config']
    exported['fit_report'] = {**exported['fit_report'], 'continuation': helper.summary(imported),
                              'legacy_reuse': imported['legacy_import']}
    atomic_save(output / 'model.pth', exported, torch_format=True)
    atomic_save(output / 'legacy-import.json', imported['legacy_import'])
    return output
