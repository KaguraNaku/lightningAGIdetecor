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
        from configs import DEFAULT_CONFIG

        cls.get(name)
        return DEFAULT_CONFIG

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
        return model.get('method')

    @classmethod
    def validate_factory(cls, model):
        factory = cls.resolve_factory(model.get('factory'))
        if factory not in {s.factory for s in cls.SPECS}:
            raise ValueError('Factory is outside the nine-model scope: '+str(factory))
        return model

    @classmethod
    def select(cls, name, model):
        spec = cls.get(name)
        result = dict(model)
        result.setdefault('factory', spec.factory)
        cls.validate_factory(result)
        canonical = cls.resolve_factory(result['factory'])
        if canonical != spec.factory:
            raise ValueError('Selected model differs from configured factory')
        result['factory'] = canonical
        return result
