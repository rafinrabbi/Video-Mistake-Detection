from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np


def median_smooth(scores: List[float], k: int = 3) -> List[float]:
    arr = np.array(scores)
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    return [float(np.median(padded[i:i+k])) for i in range(len(scores))]


def detect_start_of_deviation(
    smoothed_scores: List[float],
    timestamps: List[float],
    tau: float,
    min_consecutive: int = 2,
) -> Optional[float]:
    """First timestamp where score >= tau for at least min_consecutive clips."""
    count = 0
    start_idx: Optional[int] = None
    for i, s in enumerate(smoothed_scores):
        if s >= tau:
            if count == 0:
                start_idx = i
            count += 1
            if count >= min_consecutive:
                return timestamps[start_idx]
        else:
            count = 0
            start_idx = None
    return None


def clip_predictions_to_frame_labels(
    clip_predictions: List[Tuple[float, float, int, str]],
    total_duration_sec: float,
    fps: float = 1.0,
) -> Tuple[List[int], List[str]]:
    """
    Map per-clip (start, end, label, class) to per-frame lists.
    clip_predictions: [(start_sec, end_sec, binary_label, mistake_class), ...]
    Returns (frame_binary_labels, frame_classes)
    """
    n_frames = int(total_duration_sec * fps)
    binary = [0] * n_frames
    classes = ["no_mistake"] * n_frames
    for start, end, label, cls in clip_predictions:
        fi = int(start * fps)
        fe = min(int(end * fps), n_frames)
        for f in range(fi, fe):
            binary[f] = label
            classes[f] = cls
    return binary, classes
