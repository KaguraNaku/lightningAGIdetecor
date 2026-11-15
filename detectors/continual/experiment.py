"""Compose the native UFD adapter with matched training-state continuation."""
from copy import deepcopy
from pathlib import Path

import torch

from data.manifest import sha256
from detectors.adaptation.experiment import UFDAdaptationTask


class ContinualUFDTask(UFDAdaptationTask):
    def __init__(self, settings, *, state=None):
        self.stage_parent = settings.get('previous_stage_model') if state is None else None
        if self.stage_parent:
            ref = self.stage_parent
            if sha256(Path(ref['path'])) != ref['sha256']:
                raise ValueError('Previous stage model envelope checksum changed')
            saved = torch.load(ref['path'], weights_only=True, mmap=True, map_location='cpu')
            if saved['model']['factory'] != 'detectors.continual.experiment:create_task':
                raise ValueError('A later stage must use the prior selected continual model')
            super().__init__({}, state=saved['task'])
        else:
            super().__init__(settings, state=state)
        self.adaptation.update(regime='sequential_no_replay',
            optimizer='Native optimizer initialized once at FF++→UFD;all state restored from each selected prior epoch',
            worker_rng='Nonpersistent workers restart at epoch boundaries from carried loader generator')
        self.task.origin['UFD_adaptation'] = deepcopy(self.adaptation)

    def fit(self, *args, **kwargs):
        spec = kwargs['settings'].get('continuation')
        if not spec or ((spec['stage_index'] > 0) != bool(self.stage_parent)):
            raise ValueError('Continual fitting requires aligned explicit stage and previous-model references')
        return super().fit(*args, **kwargs)


def create_task(settings, *, state=None):
    return ContinualUFDTask(settings, state=state)
