"""Authoritative nine-model catalog; importing it loads no model backend."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: str
    fit_kind: str = 'gradient'


class ModelRegistry:
    SPECS = tuple(ModelSpec(name, 'detectors.'+name+'.experiment:create_task')
                  for name in ('xception','ucf','f3net','core','gend','effort','forada','prodet')) + (
        ModelSpec('lightning','detectors.lightning_detetcor.fixed8_task:create_task','analytic'),)
    WRAPPERS = frozenset(('detectors.adaptation.experiment:create_task',
                          'detectors.continual.experiment:create_task'))
    LEGACY_LIGHTNING = 'detectors.lightning_detetcor.experiment:create_task'

    @classmethod
    def default_config(cls, name):
        from pathlib import Path
        cls.get(name)
        filename = 'ffpp_cross_dataset.yaml' if name == 'lightning' else name + '_ffpp.yaml'
        return Path(__file__).resolve().parents[1] / 'configs' / 'experiments' / filename

    @classmethod
    def describe(cls):
        return [{'name':spec.name,'factory':spec.factory,'fit_kind':spec.fit_kind}
                for spec in cls.SPECS]

    @classmethod
    def get(cls, name):
        for spec in cls.SPECS:
            if spec.name == name:
                return spec
        raise ValueError('Unknown supported model: '+str(name))

    @classmethod
    def method_for(cls, model):
        if model.get('method'):
            return model['method']
        base = model.get('base_model', {})
        factory = base.get('factory', '')
        for spec in cls.SPECS:
            if spec.factory == factory:
                return spec.name
        return None

    @classmethod
    def validate_factory(cls, model):
        factory = model.get('factory')
        if factory in cls.WRAPPERS:
            method = cls.method_for(model)
            if method not in {s.name for s in cls.SPECS if s.fit_kind=='gradient'}:
                raise ValueError('Adaptation wrappers require one of the eight native models')
        elif factory not in {s.factory for s in cls.SPECS} | {cls.LEGACY_LIGHTNING}:
            raise ValueError('Factory is outside the nine-model scope: '+str(factory))
        return model

    @classmethod
    def select(cls, name, model):
        spec = cls.get(name)
        result = dict(model)
        result.setdefault('factory',spec.factory)
        cls.validate_factory(result)
        if result['factory'] in cls.WRAPPERS:
            if cls.method_for(result) != name:
                raise ValueError('Selected model differs from adaptation method')
        elif result['factory'] != spec.factory:
            raise ValueError('Selected model differs from configured factory')
        return result
