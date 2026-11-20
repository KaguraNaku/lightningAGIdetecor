"""Nine-model catalog and the single compatibility boundary for saved factories."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: str
    fit_kind: str = 'gradient'


class ModelRegistry:
    """Resolve historical factory names without importing any model backend.

    Saved checkpoints retain their original declarations. Dispatch resolves an
    alias to a canonical class constructor in memory; no alias modules or
    process-global module substitutions are installed.
    """

    NATIVE_CLASSES = {
        'xception': 'XceptionTask', 'ucf': 'UCFTask', 'f3net': 'F3NetTask',
        'core': 'CORETask', 'gend': 'GenDTask', 'effort': 'EffortTask',
        'forada': 'ForAdaTask', 'prodet': 'ProDetTask',
    }
    SPECS = tuple(ModelSpec(name, f'detectors.{name}.task:{class_name}')
                  for name, class_name in NATIVE_CLASSES.items()) + (
        ModelSpec('lightning', 'detectors.lightning.task:LightningTask', 'analytic'),)
    LEGACY_UFD_FACTORIES = {
        'detectors.adaptation.experiment:create_task': 'adaptation',
        'detectors.adaptation.task:UFDAdaptationTask': 'adaptation',
        'engine.adaptation:UFDAdaptationTask': 'adaptation',
        'detectors.continual.experiment:create_task': 'continual',
        'detectors.continual.task:ContinualUFDTask': 'continual',
        'engine.continual:ContinualUFDTask': 'continual',
    }
    FACTORY_ALIASES = {
        **{f'detectors.{name}.experiment:create_task': f'detectors.{name}.task:{class_name}'
           for name, class_name in NATIVE_CLASSES.items()},
    }

    @classmethod
    def resolve_factory(cls, factory):
        """Canonicalize known names; preserve unrelated engine test adapters."""
        if not isinstance(factory, str) or ':' not in factory:
            raise ValueError('Factory must be module:callable')
        return cls.FACTORY_ALIASES.get(factory, factory)

    @classmethod
    def default_config(cls, name):
        from pathlib import Path
        cls.get(name)
        return Path(__file__).resolve().parents[1] / 'configs' / 'experiments' / (name + '_ffpp.yaml')

    @classmethod
    def describe(cls):
        return [{'name':spec.name, 'factory':spec.factory, 'fit_kind':spec.fit_kind}
                for spec in cls.SPECS]

    @classmethod
    def get(cls, name):
        for spec in cls.SPECS:
            if spec.name == name:
                return spec
        raise ValueError('Unknown supported model: '+str(name))

    @classmethod
    def method_for(cls, model):
        factory = model.get('factory')
        canonical = cls.resolve_factory(factory) if factory else None
        for spec in cls.SPECS:
            if canonical == spec.factory:
                return spec.name
        if canonical in cls.LEGACY_UFD_FACTORIES:
            return model.get('method') or cls.method_for(model.get('base_model', {}))
        return model.get('method')

    @classmethod
    def validate_factory(cls, model):
        factory = cls.resolve_factory(model.get('factory'))
        if factory in cls.LEGACY_UFD_FACTORIES:
            method = cls.method_for(model)
            if method not in cls.NATIVE_CLASSES:
                raise ValueError('Adaptation wrappers require one of the eight native models')
            base = model.get('base_model')
            if base is not None:
                cls.validate_factory(base)
                if cls.resolve_factory(base['factory']) != cls.get(method).factory:
                    raise ValueError('Adaptation method differs from its base model')
        elif factory not in {s.factory for s in cls.SPECS}:
            raise ValueError('Factory is outside the nine-model scope: '+str(factory))
        return model

    @classmethod
    def select(cls, name, model):
        spec = cls.get(name)
        result = dict(model)
        result.setdefault('factory', spec.factory)
        cls.validate_factory(result)
        canonical = cls.resolve_factory(result['factory'])
        if canonical in cls.LEGACY_UFD_FACTORIES:
            if cls.method_for(result) != name:
                raise ValueError('Selected model differs from adaptation method')
        elif canonical != spec.factory:
            raise ValueError('Selected model differs from configured factory')
        result['factory'] = canonical
        return result
