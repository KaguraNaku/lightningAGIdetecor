"""Shared full-source isolation checks applied before STIL clip exclusions."""

from detectors.i3d.protocol import validate_source_splits as validate_full_source


def validate_source_splits(train, validation):
    try:
        result = validate_full_source(train, validation)
    except ValueError as error:
        raise ValueError(f"STIL source isolation: {error}") from error
    return {**result, "method": "stil", "clip_size": 8}
