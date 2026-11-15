"""Explicit local resource resolution without model imports or global env writes."""

from copy import deepcopy
import os
from pathlib import Path
import re


class ResourceResolver:
    """Expand declared variables and anchor local files to their configuration.

    Hub identifiers are not filesystem paths. Only declared resource fields are
    interpreted as paths; thresholds, names and checkpoint provenance stay data.
    Saved checkpoint payloads are never rewritten by this resolver.
    """

    VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    FILE_FIELDS = frozenset({
        "init_weights", "initial_checkpoint", "base_clip_checkpoint",
        "checkpoint", "mapping_path", "coefficients_path", "statistics_path",
        "parent_checkpoint",
    })
    DIRECTORY_FIELDS = frozenset({"cache_dir", "assets_dir", "model_dir"})

    def __init__(self, base_dir, *, variables=None, env_file=None):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.env_file = Path(env_file).expanduser().resolve() if env_file else None
        declared = {}
        if self.env_file is not None:
            if not self.env_file.is_file():
                raise FileNotFoundError(f"Environment file not found: {self.env_file}")
            from dotenv import dotenv_values

            declared = dotenv_values(self.env_file, interpolate=False)
        self.variables = {**declared, **os.environ, **(variables or {})}

    def expand(self, value, stack=()):
        if not isinstance(value, str):
            raise TypeError("Resource values must be strings")

        def replace(match):
            name = match[1]
            if name in stack:
                raise ValueError(f"Cyclic environment reference: {name}")
            resolved = self.variables.get(name)
            if not isinstance(resolved, str) or not resolved:
                raise ValueError(f"Missing environment variable {name}")
            return self.expand(resolved, (*stack, name))

        expanded = self.VARIABLE.sub(replace, value)
        if "$" in expanded:
            raise ValueError(f"Unsupported environment expression: {value}; use ${{NAME}}")
        return expanded

    def path(self, value, *, must_exist=True, kind="file"):
        if kind not in {"file", "directory", "any"}:
            raise ValueError("Resource kind must be file, directory or any")
        expanded = self.expand(str(value))
        if not expanded:
            raise ValueError("Resource path must not be empty")
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        if must_exist:
            valid = path.is_file() if kind == "file" else path.is_dir() if kind == "directory" else path.exists()
            if not valid:
                raise FileNotFoundError(f"Required {kind} resource not found: {path}")
        return path

    def model_settings(self, settings):
        """Resolve input assets in a copied model declaration, including parents."""
        def visit(value):
            if isinstance(value, list):
                return [visit(item) for item in value]
            if not isinstance(value, dict):
                return deepcopy(value)
            result = {}
            for key, item in value.items():
                if key == "statistics" and isinstance(item, dict):
                    result[key] = {
                        branch: str(self.path(source)) if isinstance(source, str) else visit(source)
                        for branch, source in item.items()
                    }
                elif isinstance(item, str) and key in self.FILE_FIELDS | {"path"}:
                    result[key] = str(self.path(item))
                elif isinstance(item, str) and key in self.DIRECTORY_FIELDS:
                    result[key] = str(self.path(item, kind="directory"))
                elif isinstance(item, str) and key == "backbone_id":
                    expanded = self.expand(item)
                    result[key] = str(self.path(expanded, kind="directory")) if expanded.startswith(("/", "~", ".")) else expanded
                else:
                    result[key] = visit(item)
            return result

        return visit(settings)
