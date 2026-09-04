from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np

try:
    import decord
    decord.bridge.set_bridge("torch")
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False


@dataclass
class Clip:
    clip_id: str
    task_id: str
    start_sec: float
    end_sec: float
    frames: "np.ndarray | None" = None   # (T, H, W, C) uint8, loaded lazily
    true_label: int = 0                  # 1 = mistake in this clip
    true_class: str = "no_mistake"


def _load_frames(video_path: Path, start_sec: float, end_sec: float, fps: float) -> "np.ndarray":
    if not _DECORD_AVAILABLE:
        raise RuntimeError("decord not installed")
    vr = decord.VideoReader(str(video_path))
    native_fps = vr.get_avg_fps()
    start_f = int(start_sec * native_fps)
    end_f = min(int(end_sec * native_fps), len(vr) - 1)
    step = max(1, int(native_fps / fps))
    indices = list(range(start_f, end_f, step))
    if not indices:
        indices = [start_f]
    frames = vr.get_batch(indices)
    if hasattr(frames, "asnumpy"):
        return frames.asnumpy()   # (T, H, W, C)
    return frames.cpu().numpy() if hasattr(frames, "cpu") else np.asarray(frames)


def online_clips(
    video_path: Path,
    clip_id: str,
    task_id: str,
    duration_sec: float,
    annotations: list,
    clip_len_sec: float = 4.0,
    stride_sec: float = 2.0,
    fps: float = 1.0,
    load_frames: bool = False,
) -> Generator[Clip, None, None]:
    """Causal sliding window — no future access."""
    t = 0.0
    while t + clip_len_sec <= duration_sec:
        start, end = t, t + clip_len_sec
        mid = (start + end) / 2
        label = int(any(a.start_sec <= mid <= a.end_sec for a in annotations))
        cls = "no_mistake"
        for a in annotations:
            if a.start_sec <= mid <= a.end_sec:
                cls = a.mistake_class
                break
        frames = _load_frames(video_path, start, end, fps) if load_frames else None
        yield Clip(clip_id=clip_id, task_id=task_id, start_sec=start, end_sec=end,
                   frames=frames, true_label=label, true_class=cls)
        t += stride_sec


def offline_clips(
    video_path: Path,
    clip_id: str,
    task_id: str,
    duration_sec: float,
    annotations: list,
    clip_len_sec: float = 4.0,
    stride_sec: float = 2.0,
    fps: float = 1.0,
    load_frames: bool = False,
) -> List[Clip]:
    """Full-video access — all clips at once."""
    return list(online_clips(
        video_path, clip_id, task_id, duration_sec, annotations,
        clip_len_sec, stride_sec, fps, load_frames,
    ))
