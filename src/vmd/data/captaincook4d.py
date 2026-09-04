"""CaptainCook4D loader — emits the legacy ``EgoOopsClip`` interface.

CaptainCook4D is the sole benchmark for this project. See
``data/raw/captaincook4d/INSPECTION.md`` for the verified schema and counts,
and ``mistake_classes.py`` for the native 8-class taxonomy (CC4D's 8
categories minus ``Other``).

Layout expected (annotations cloned from CaptainCook4D/annotations):

    <root>/captaincook4d/
      annotations/
        annotation_json/{error_annotations,activity_idx_step_idx}.json
        annotation_csv/activity_step_description.csv
        data_splits/recordings_data_split_combined.json
        metadata/video_information.csv
        task_graphs/<recipe>.json
      videos/<recording_id>.*          # downloaded separately (--data2d)

Key per-dataset quirks (handled here):
* recording_id is "<activity_id>_<n>" (e.g. "1_10" -> activity 1).
* recipe/task_id = activity_name lowercased, alphanumerics only
  ("Microwave Egg Sandwich" -> "microwaveeggsandwich") which is also the
  task_graphs/<recipe>.json filename, so graphs line up with clip.task_id.
* 281 error segments have start_time == -1 (Missing-Step etc. never happened):
  non-localizable. Kept with start=end=-1 (so time-window `is_mistake` skips
  them) unless `drop_nonlocalizable=True`. They still matter for segment-level
  metrics — count is logged on load.
* The ``Other`` tag (8 instances) is the only category dropped from the typed
  metric (counted in ``untyped_dropped``); a step whose only tags are ``Other``
  becomes ``no_mistake`` for typed eval but still counts toward binary mistake.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vmd.data.egoops import EgoOopsClip, MistakeAnnotation, StepSpan
from vmd.data.mistake_classes import step_label


def recipe_key(activity_name: str) -> str:
    """Normalise an activity name to its task_graphs/<key>.json stem."""
    return re.sub(r"[^a-z0-9]", "", activity_name.lower())


def _activity_id_of(recording_id: str) -> int:
    return int(recording_id.split("_")[0])


def _load_activity_meta(
    ann_root: Path,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Return (activity_id -> name, activity_id -> numbered task_description)."""
    csv_path = ann_root / "annotation_csv" / "activity_step_description.csv"
    names: Dict[int, str] = {}
    # activity_id -> {step_index: description}
    steps: Dict[int, Dict[int, str]] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            aid = int(r["activity_idx"])
            names[aid] = r["activity_name"]
            steps.setdefault(aid, {})[int(r["step_index"])] = r["step_description"]

    # Order steps by the canonical sequence (first occurrence), de-duplicated.
    seq_path = ann_root / "annotation_json" / "activity_idx_step_idx.json"
    canonical = json.loads(seq_path.read_text())  # {activity_id(str): [step_idx,...]}
    descriptions: Dict[int, str] = {}
    for aid, step_map in steps.items():
        order = canonical.get(str(aid), sorted(step_map))
        seen: List[int] = []
        for s in order:
            if s in step_map and s not in seen:
                seen.append(s)
        # include any steps not present in the canonical sequence
        for s in sorted(step_map):
            if s not in seen:
                seen.append(s)
        descriptions[aid] = "\n".join(
            f"{i}. {step_map[s]}" for i, s in enumerate(seen, start=1)
        )
    return names, descriptions


def _load_durations(ann_root: Path) -> Dict[str, float]:
    path = ann_root / "metadata" / "video_information.csv"
    out: Dict[str, float] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["recording_id"]] = float(r["duration(sec)"])
    return out


def _load_splits(ann_root: Path, combined: bool = True) -> Dict[str, str]:
    """Return recording_id -> split ('train'|'val'|'test')."""
    name = "recordings_data_split_combined.json" if combined else "recordings_data_split_normal.json"
    data = json.loads((ann_root / "data_splits" / name).read_text())
    rid_split: Dict[str, str] = {}
    for split, rids in data.items():
        for rid in rids:
            rid_split[rid] = split
    return rid_split


def _find_video(videos_dir: Path, recording_id: str) -> Path:
    """Resolve the video file for a recording, tolerant of naming variants.

    The 2D downloader writes ``<rid>_360p.mp4``. Glob ``<rid>.*`` and
    ``<rid>_*`` (NOT bare ``<rid>*``: ids are prefix-ambiguous — "12_2*"
    would also match 12_26_360p.mp4) and fall back to ``<rid>.mp4`` so the
    loader works before the videos land too.
    """
    if videos_dir.exists():
        matches = sorted(videos_dir.glob(f"{recording_id}.*")) or sorted(
            videos_dir.glob(f"{recording_id}_*")
        )
        if matches:
            return matches[0]
    return videos_dir / f"{recording_id}.mp4"


class CaptainCook4DDataset:
    """CaptainCook4D loader, exposing ``.clips`` as ``List[EgoOopsClip]``.

    Args:
        raw_dir: ``.../data/raw/captaincook4d`` (contains ``annotations/`` and
            optionally ``videos/``).
        split: 'train' | 'val' | 'test'.
        combined_split: use the combined (normal+error) split (default True).
        drop_nonlocalizable: drop start=-1 error segments entirely.
    """

    def __init__(
        self,
        raw_dir: Path,
        split: str = "train",
        combined_split: bool = True,
        drop_nonlocalizable: bool = False,
        videos_dir: Optional[Path] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.ann_root = self.raw_dir / "annotations"
        self.split = split
        self.videos_dir = Path(videos_dir) if videos_dir else self.raw_dir / "videos"
        self.clips: List[EgoOopsClip] = []
        # load stats
        self.nonlocalizable = 0
        self.untyped_dropped = 0
        self._load(split, combined_split, drop_nonlocalizable)

    def _load(self, split: str, combined: bool, drop_nonloc: bool) -> None:
        names, descriptions = _load_activity_meta(self.ann_root)
        durations = _load_durations(self.ann_root)
        rid_split = _load_splits(self.ann_root, combined)
        err = json.loads(
            (self.ann_root / "annotation_json" / "error_annotations.json").read_text()
        )
        err_by_rid = {r["recording_id"]: r for r in err}
        # Gold step segmentation: the real answer to "which step is the person
        # on", which the window-index heuristic was standing in for.
        step_path = self.ann_root / "annotation_json" / "step_annotations.json"
        steps_by_rid = json.loads(step_path.read_text()) if step_path.exists() else {}

        # every recording that belongs to this split (error + normal)
        for rid, sp in rid_split.items():
            if sp != split:
                continue
            aid = _activity_id_of(rid)
            name = names.get(aid, f"activity_{aid}")
            rec = err_by_rid.get(rid)
            annotations: List[MistakeAnnotation] = []
            if rec is not None and rec.get("is_error"):
                for st in rec.get("step_annotations", []):
                    errors = st.get("errors", [])
                    if not errors:
                        continue
                    cls = step_label(errors)
                    if cls == "no_mistake":
                        # only unmapped tags (``Other``) on this step
                        self.untyped_dropped += 1
                        continue
                    start = float(st.get("start_time", -1.0))
                    end = float(st.get("end_time", -1.0))
                    if start < 0 or end < 0:
                        self.nonlocalizable += 1
                        if drop_nonloc:
                            continue
                        start, end = -1.0, -1.0
                    desc = "; ".join(
                        e.get("description", "").strip()
                        for e in errors
                        if e.get("description", "").strip()
                    )
                    annotations.append(
                        MistakeAnnotation(
                            start_sec=start,
                            end_sec=end,
                            mistake_class=cls,
                            description=desc,
                        )
                    )
            spans = [
                StepSpan(
                    step_id=int(st.get("step_id", -1)),
                    start_sec=float(st.get("start_time", -1.0)),
                    end_sec=float(st.get("end_time", -1.0)),
                    description=str(st.get("description", "")),
                )
                for st in steps_by_rid.get(rid, {}).get("steps", [])
            ]
            spans.sort(key=lambda sp: (sp.start_sec if sp.start_sec >= 0 else 1e9))
            self.clips.append(
                EgoOopsClip(
                    clip_id=rid,
                    task_id=recipe_key(name),
                    task_name=name,
                    task_description=descriptions.get(aid, ""),
                    video_path=_find_video(self.videos_dir, rid),
                    duration_sec=durations.get(rid, 0.0),
                    annotations=annotations,
                    step_spans=spans,
                    split=split,
                )
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> EgoOopsClip:
        return self.clips[idx]

    def by_task(self) -> Dict[str, List[EgoOopsClip]]:
        out: Dict[str, List[EgoOopsClip]] = {}
        for c in self.clips:
            out.setdefault(c.task_id, []).append(c)
        return out

    def task_descriptions(self) -> Dict[str, Tuple[str, str]]:
        seen: Dict[str, Tuple[str, str]] = {}
        for c in self.clips:
            if c.task_id not in seen:
                seen[c.task_id] = (c.task_name, c.task_description)
        return seen
