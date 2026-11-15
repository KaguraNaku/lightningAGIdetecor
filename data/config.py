"""Explicit .env loading; dataset paths are resolved only when requested."""

import os
from importlib.resources import files
from pathlib import Path

from dotenv import dotenv_values
import yaml

from engine.resources import ResourceResolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataConfig:
    def __init__(
        self, path: Path | str | None = None, env_file: Path | str | None = None
    ):
        self.path = Path(
            path or str(files("configs").joinpath("datasets.yaml"))
        ).resolve()
        local_root = (
            PROJECT_ROOT if (PROJECT_ROOT / "pyproject.toml").is_file() else Path.cwd()
        )
        self.env_file = Path(env_file or local_root / ".env")
        # Read without mutating process-wide environment. Explicit env wins,
        # including for references inside .env values.
        self.variables = {
            **dotenv_values(self.env_file, interpolate=False),
            **os.environ,
        }
        self.resources = ResourceResolver(self.path.parent, variables=self.variables)
        self.variables = self.resources.variables
        with self.path.open() as stream:
            self.settings = yaml.safe_load(stream)
        if not isinstance(self.settings, dict) or not isinstance(
            self.settings.get("datasets"), dict
        ):
            raise ValueError(f"{self.path}: datasets must be a mapping")

    def expand(self, value: str, stack: tuple[str, ...] = ()) -> str:
        return self.resources.expand(value, stack)

    def dataset(self, name: str) -> dict:
        if name not in self.settings["datasets"]:
            raise ValueError(f"Unknown dataset: {name}")
        settings = dict(self.settings["datasets"][name])
        root = Path(self.expand(settings["root"])).expanduser()
        if not root.is_absolute():
            raise ValueError(f"{name}: dataset root must be absolute")
        if not root.is_dir():
            raise FileNotFoundError(f"{name}: dataset root not found: {root}")
        settings["root"] = root.resolve()
        return settings

    def manifest_dir(self) -> Path:
        value = self.variables.get(
            "MANIFEST_ROOT", "~/.local/share/lightning-remake/manifests"
        )
        path = Path(self.expand(value)).expanduser()
        if not path.is_absolute():
            raise ValueError("MANIFEST_ROOT must be absolute")
        return path
