"""Reusable source/video sampling contracts, checked against actual manifests."""
from collections import Counter


class VideoSamplingPolicy:
    """Eight source frames per video; at most 32 for validation and testing.

    Image-only domains have no video sampling requirement. No rows are removed,
    resampled or repeated to make a mismatching manifest pass.
    """
    NAME = "train8_test32"

    def __init__(self, settings):
        if settings != {"name": self.NAME, "train_frames": 8, "test_frames": 32}:
            raise ValueError("Video sampling must be strictly train8_test32")

    def validate(self, header, records):
        sampling = header.get("sampling", {})
        if sampling.get("limit") is not None:
            raise ValueError("Reported experiments cannot use a smoke-test manifest")
        counts = Counter(row.video_id for row in records if row.video_id)
        if not counts:
            if header["dataset"] == "FaceForensics++":
                raise ValueError("FF++ requires explicit video identities")
            return {"policy": self.NAME, "unit": "image", "images": len(records)}
        if sum(counts.values()) != len(records):
            raise ValueError("Video manifest mixes missing and present video identities")
        training = header["split"] == "train"
        frames = 8 if training else 32
        if sampling.get("frames_per_video") != frames:
            raise ValueError(f"Manifest must declare {frames} frames per video")
        if any(n != 8 if training else n > 32 for n in counts.values()):
            raise ValueError("Actual video frame counts violate train8_test32")
        return {"policy": self.NAME, "unit": "video", "videos": len(counts),
                "images": len(records), "min_frames": min(counts.values()),
                "max_frames": max(counts.values())}
