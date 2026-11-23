"""On-demand access to the version-bound native augmentation library."""

from functools import cache


@cache
def load_albumentations():
    """Import Albumentations only when a selected training recipe needs it."""
    import albumentations

    required_version = "1.3.1"
    if albumentations.__version__ != required_version:
        raise RuntimeError(
            f"Augmentation requires albumentations=={required_version}, "
            f"found {albumentations.__version__}"
        )
    return albumentations
