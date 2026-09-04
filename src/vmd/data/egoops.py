"""Legacy clip dataclasses retained for downstream compatibility.

The EgoOops dataset itself has been removed from this project (CC4D-only
benchmark). The ``EgoOopsClip`` / ``MistakeAnnotation`` names live on as the
universal clip schema consumed by samplers, eval loops, and training datasets.

Canonical mistake-class taxonomy lives in ``vmd.data.mistake_classes``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class MistakeAnnotation:
    start_sec: float
    end_sec: float
    mistake_class: str
    description: str = ""


@dataclass
class StepSpan:
    """One gold procedure step with its time bounds.

    CaptainCook4D ships these (``step_annotations.json``) and the project had
    never loaded them: the step shown to the model was derived from the window
    index instead, which is what made the conditioning degenerate.
    ``start_sec < 0`` marks a step with no localizable bounds (a missing step).
    """
    step_id: int
    start_sec: float
    end_sec: float
    description: str = ""


@dataclass
class EgoOopsClip:
    clip_id: str
    task_id: str
    task_name: str
    task_description: str
    video_path: Path
    duration_sec: float
    annotations: List[MistakeAnnotation] = field(default_factory=list)
    step_spans: List["StepSpan"] = field(default_factory=list)
    split: str = "train"

    def is_mistake(self, t_sec: float) -> bool:
        return any(a.start_sec <= t_sec <= a.end_sec for a in self.annotations)

    def mistake_class_at(self, t_sec: float) -> str:
        for a in self.annotations:
            if a.start_sec <= t_sec <= a.end_sec:
                return a.mistake_class
        return "no_mistake"
