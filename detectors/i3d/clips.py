"""Native T16 clip selection over explicitly complete, ordered video pools.

This module uses the standard library only. Completeness must be established by
the manifest builder/adapter; the flag is a required contract, not a disk scan.
Validation covers the entire supplied pool before the native first32 truncation.
"""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class VideoFrames:
    dataset: str
    split: str
    video_id: str
    frames: tuple
    complete_pool: bool

    def __post_init__(self):
        if self.complete_pool is not True:
            raise ValueError("I3D requires an explicitly complete source frame pool")
        if not self.frames:
            raise ValueError("I3D source video frame pools cannot be empty")
        if not all(
            isinstance(value, str) and value
            for value in (self.dataset, self.split, self.video_id)
        ):
            raise ValueError("Video pool dataset/split/video identity is required")
        labels, ids, positions = set(), set(), set()
        for frame in self.frames:
            if (frame.dataset, frame.split, frame.video_id) != (
                self.dataset,
                self.split,
                self.video_id,
            ):
                raise ValueError("Source frame identity differs from its video pool")
            if type(frame.label) is not int or frame.label not in {0, 1}:
                raise ValueError("Source frame labels must be binary")
            if type(frame.frame_index) is not int or frame.frame_index < 0:
                raise ValueError("I3D needs physical source frame indices")
            if frame.id in ids or frame.frame_index in positions:
                raise ValueError("Duplicate source frame ID or physical position")
            ids.add(frame.id)
            positions.add(frame.frame_index)
            labels.add(frame.label)
        if len(labels) != 1:
            raise ValueError("Conflicting source labels within a video pool")
        object.__setattr__(
            self,
            "frames",
            tuple(sorted(self.frames, key=lambda frame: frame.frame_index)),
        )


@dataclass(frozen=True)
class ClipRecord:
    id: str
    dataset: str
    split: str
    label: int
    video_id: str
    frames: tuple
    source_frame_ids: tuple
    frame_positions: tuple
    pool_start: int
    storage: str = "clip"


def select_clips(videos, *, training, rng):
    """Select once at dataset construction, then shuffle even during evaluation.

    Supply an explicit random.Random or the random module. A private generator
    preserves native selection/draw order conditional on its initial state;
    it does not claim native global process/worker RNG equivalence.
    """
    if type(training) is not bool:
        raise ValueError("training must be boolean")
    clips, inventories, seen = [], [], set()
    for video in videos:
        key = video.dataset, video.split, video.video_id
        if key in seen:
            raise ValueError("Duplicate video pool identity")
        seen.add(key)
        frames, available = video.frames, len(video.frames)
        inventory = {
            "dataset": video.dataset,
            "split": video.split,
            "video_id": video.video_id,
            "available_frames": available,
            "source_frame_ids": [f.id for f in frames],
            "excluded_reason": "shorter_than_clip16" if available < 16 else None,
        }
        inventories.append(inventory)
        if available > 32:
            # This apparently redundant draw is part of native training RNG.
            start = rng.randint(0, 0) if training else 0
            frames = frames[start : start + 32]
        count = len(frames)
        if count < 16:
            continue
        if count // 16 > 1:
            starts = [
                rng.randrange(0, 16) if training else 0,
                rng.randrange(16, 17) if training else 16,
            ]
        else:
            starts = [rng.randrange(0, count - 16 + 1) if training else 0]
        for start in starts:
            selected = frames[start : start + 16]
            identity = {"video": key, "frame_ids": [frame.id for frame in selected]}
            clip_id = (
                "i3d-clip-"
                + hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()
            )
            clips.append(
                ClipRecord(
                    id=clip_id,
                    dataset=video.dataset,
                    split=video.split,
                    label=selected[0].label,
                    video_id=video.video_id,
                    frames=selected,
                    source_frame_ids=tuple(frame.id for frame in video.frames),
                    frame_positions=tuple(frame.frame_index for frame in selected),
                    pool_start=start,
                )
            )
    if clips:
        rng.shuffle(clips)
    return clips, {
        "selection": "native_first32_T16_construction_time_then_shuffle",
        "training": training,
        "source_videos": inventories,
        "selected_clips": len(clips),
        "source_frame_pool_completeness_independently_verified": False,
        "completeness_contract": "Required from manifest builder/adapter, not established by this selection primitive",
    }
