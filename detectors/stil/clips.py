"""Native STIL T8 selection over validated complete source snapshots."""

import hashlib
import json

from detectors.i3d.clips import ClipRecord, VideoFrames


def select_clips(videos, *, training, rng):
    if type(training) is not bool:
        raise ValueError("STIL training flag must be boolean")
    clips, inventories, seen = [], [], set()
    for video in videos:
        if not isinstance(video, VideoFrames):
            raise ValueError("STIL selection requires validated complete VideoFrames")
        key = video.dataset, video.split, video.video_id
        if key in seen:
            raise ValueError("Duplicate STIL source video identity")
        seen.add(key)
        frames = video.frames
        inventories.append(
            {
                "dataset": video.dataset,
                "split": video.split,
                "video_id": video.video_id,
                "available_frames": len(frames),
                "source_frame_ids": [frame.id for frame in frames],
                "excluded_reason": "shorter_than_clip8" if len(frames) < 8 else None,
            }
        )
        if len(frames) > 32:
            # Original total_frames is already reset to32 before this draw.
            start = rng.randint(0, 0) if training else 0
            frames = frames[start : start + 32]
        count = len(frames)
        if count < 8:
            continue
        number = count // 8
        if number > 1:
            step = (count - 8) // (number - 1)
            starts = [
                rng.randrange(i * step, min((i + 1) * step, count - 8 + 1))
                if training
                else i * step
                for i in range(number)
            ]
        else:
            starts = [rng.randrange(0, count - 8 + 1) if training else 0]
        for start in starts:
            selected = frames[start : start + 8]
            identity = {"video": key, "frame_ids": [frame.id for frame in selected]}
            clip_id = (
                "stil-clip-"
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
        "selection": "native_first32_T8_construction_time_then_shuffle",
        "training": training,
        "source_videos": inventories,
        "selected_clips": len(clips),
        "source_frame_pool_completeness_independently_verified": False,
        "completeness_contract": "All rows of bound preparation snapshot;not a fresh filesystem/author-corpus completeness claim",
    }
