"""Explicit .env loading; dataset paths are resolved only when requested."""

import os
from pathlib import Path

from dotenv import dotenv_values
import yaml

from configs import DEFAULT_CONFIG
from engine.resources import ResourceResolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_env_file():
    root = PROJECT_ROOT if (PROJECT_ROOT / "pyproject.toml").is_file() else Path.cwd()
    return root / ".env"


def frames_for(split, settings):
    count = settings.get("train_frames" if split == "train" else "test_frames")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("Video frame counts must be positive integers")
    return count


class DataConfig:
    @classmethod
    def for_experiment(cls, path, settings, *, registry=None, env_file=None):
        if registry is None and "datasets" in settings:
            return cls(path, env_file, settings=settings)
        return cls(registry, env_file)

    def __init__(
        self, path: Path | str | None = None, env_file: Path | str | None = None, *, settings=None
    ):
        self.path = Path(path or DEFAULT_CONFIG).resolve()
        self.env_file = Path(env_file or default_env_file())
        # Read without mutating process-wide environment. Explicit env wins,
        # including for references inside .env values.
        self.variables = {
            **dotenv_values(self.env_file, interpolate=False),
            **os.environ,
        }
        self.resources = ResourceResolver(self.path.parent, variables=self.variables)
        self.variables = self.resources.variables
        if settings is None:
            with self.path.open() as stream:
                settings = yaml.safe_load(stream)
        self.settings = settings
        if not isinstance(self.settings, dict) or not isinstance(
            self.settings.get("datasets"), dict
        ):
            raise ValueError(f"{self.path}: datasets must be a mapping")

    def expand(self, value: str, stack: tuple[str, ...] = ()) -> str:
        return self.resources.expand(value, stack)

    def media_type(self, name):
        if name not in self.settings["datasets"]:
            raise ValueError(f"Unknown dataset: {name}")
        media_type = self.settings["datasets"][name].get("media_type")
        if media_type not in (None, "image", "video"):
            raise ValueError(f"{name}.media_type must be image or video")
        return media_type

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
