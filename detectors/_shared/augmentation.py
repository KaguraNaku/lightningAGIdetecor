"""On-demand access to the version-bound native augmentation library."""


class AugmentationLibrary:
    """Import Albumentations only when a selected training recipe needs it."""

    @staticmethod
    def load():
        import albumentations

        if albumentations.__version__ != "1.3.1":
            raise RuntimeError("Native synthesis requires albumentations==1.3.1")
        return albumentations
