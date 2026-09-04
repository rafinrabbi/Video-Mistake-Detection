"""MistSense — dataset.

Paper trains on Epic-tent-O / HoloAssist (§4.1). This project (VMD) ships
EgoOops; the loader is dataset-agnostic — it just reads per-clip features
produced by `scripts/extract_features.py`.

Per-clip on-disk layout expected by `MistakeClipDataset`:

    data_root/
      <clip_id>/
        visual.npy        # (T, d_visual)  — ViT-L + base Q-Former features @ 2 fps (§4.6)
        hand.npy          # (c, T, v, m)   — Mediapipe hand keypoints (§4.6)
        labels.npy        # (T,) binary    — 1 = mistake, 0 = correct (§3.3)
        explanation.txt   # ground-truth explanation text (§3.2)

Extracting these from raw video is out of scope for the minimal
reproduction; see Mediapipe / BLIP-2 docs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ClipSample:
    visual: torch.Tensor    # (t_s, d_1)
    hand: torch.Tensor      # (c, t_s, v, m)
    labels: torch.Tensor    # (t_q,) — per-query target. For minimal mode we
                            # broadcast a single segment-level label across
                            # t_q queries (per-query loss as defined in §3.3
                            # treats each query as a sample).
    explanation: str        # ground-truth explanation
    clip_id: str


class MistakeClipDataset(Dataset):
    """§4.1 — generic loader for Epic-tent-O and HoloAssist precomputed
    features. Online evaluation slides a fixed-length window over each clip;
    here we return whole segments as items.

    Args:
        data_root: directory containing per-clip subdirectories
        segment_length: §3.1 — `t_s`
        num_queries:    §3.1 — `t_q`
        split: 'train' | 'val' | 'test'
    """

    def __init__(
        self,
        data_root: str | Path,
        segment_length: int,
        num_queries: int,
        split: str = "train",
        clip_ids: Optional[list[str]] = None,
        annot_file: Optional[str | Path] = None,
        windows_per_clip: int = 8,
    ):
        self.root = Path(data_root)
        self.t_s = segment_length
        self.t_q = num_queries
        self.split = split
        if clip_ids is not None:
            self.clip_ids = clip_ids
        elif annot_file is not None:
            import json
            entries = json.loads(Path(annot_file).read_text())
            self.clip_ids = [e["clip_id"] for e in entries if (self.root / e["clip_id"]).exists()]
        else:
            self.clip_ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        # Pre-build (clip, start) windows for sliding-window sampling.
        self.windows: list[tuple[str, int]] = []
        for cid in self.clip_ids:
            labels_p = self.root / cid / "labels.npy"
            if not labels_p.exists():
                continue
            import numpy as _np
            T = int(_np.load(labels_p, mmap_mode="r").shape[0])
            n = max(1, min(windows_per_clip, T // self.t_s))
            stride = max(1, (T - self.t_s) // max(1, n - 1) if n > 1 else 1)
            for i in range(n):
                self.windows.append((cid, min(i * stride, max(0, T - self.t_s))))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        cid, start = self.windows[idx]
        d = self.root / cid
        visual = torch.from_numpy(np.load(d / "visual.npy")).float()      # (T, d_1)
        hand = torch.from_numpy(np.load(d / "hand.npy")).float()          # (c, T, v, m)
        labels = torch.from_numpy(np.load(d / "labels.npy")).float()      # (T,)
        T = visual.size(0)
        end = start + self.t_s
        if end > T:
            # pad on right
            pad_v = end - T
            visual = torch.nn.functional.pad(visual, (0, 0, 0, pad_v))
            hand = torch.nn.functional.pad(hand, (0, 0, 0, 0, 0, pad_v))
            labels = torch.nn.functional.pad(labels, (0, pad_v))
        visual = visual[start:end]
        hand = hand[:, start:end]
        seg_label = (labels[start:end].max() > 0.5).float()
        target = seg_label.repeat(self.t_q)
        explanation = (d / "explanation.txt").read_text().strip() if (d / "explanation.txt").exists() else ""
        return {
            "visual": visual,
            "hand": hand,
            "target": target,
            "explanation": explanation,
            "clip_id": cid,
        }
