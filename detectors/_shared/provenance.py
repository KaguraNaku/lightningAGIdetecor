"""Pinned reference identities, never imports code from a legacy checkout."""

import json
from importlib.resources import files


def reference_for(method):
    path = files("configs").joinpath("migrations/source_manifest.json")
    manifest = json.loads(path.read_text())
    return {
        "method": method,
        "sources": manifest["methods"][method],
        "image_pipeline_sources": manifest["methods"]["common_image_pipeline"],
    }
