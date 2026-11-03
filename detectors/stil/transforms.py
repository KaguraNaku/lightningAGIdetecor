"""STIL shares the native no-landmark pixel recipe;train AND val augment."""

from detectors.i3d.transforms import I3DClipTransform


class STILClipTransform(I3DClipTransform):
    def __init__(self, *, augment):
        if type(augment) is not bool:
            raise ValueError("STIL augmentation flag must be boolean")
        super().__init__(training=augment)
