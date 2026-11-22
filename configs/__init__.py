"""Versioned dataset and experiment configuration templates."""
from importlib.resources import files
from pathlib import Path


DEFAULT_CONFIG = Path(str(files(__name__).joinpath("config.yaml")))
