"""Full-source split isolation with explicit agreement on the TALL data recipe."""

from detectors.i3d.protocol import validate_source_splits as validate_full_source


def validate_source_splits(train, validation):
    if train.profile != validation.profile:
        raise ValueError("TALL train/validation data recipes must agree")
    result = validate_full_source(train, validation)
    return {**result, "method": "tall", "clip_size": 4, "data_profile": train.profile}
