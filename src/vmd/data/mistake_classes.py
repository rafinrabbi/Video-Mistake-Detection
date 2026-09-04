"""Single source of truth for the VMD mistake taxonomy.

Native CaptainCook4D categories with the ``Other`` bucket dropped (8 instances
total — see ``data/raw/captaincook4d/INSPECTION.md``). Class strings are
snake_case so they round-trip cleanly through VLM JSON outputs.
"""
from __future__ import annotations

from typing import Optional


MISTAKE_CLASSES: tuple[str, ...] = (
    "no_mistake",
    "order_error",
    "technique_error",
    "preparation_error",
    "measurement_error",
    "missing_step",
    "timing_error",
    "temperature_error",
)


# Maps a raw CaptainCook4D ``tag`` (as it appears in error_annotations.json)
# to its VMD class. ``None`` means "drop from typed metric" — currently only
# the ``Other`` catch-all.
CC4D_TAG_TO_CLASS: dict[str, Optional[str]] = {
    "Order Error":       "order_error",
    "Technique Error":   "technique_error",
    "Preparation Error": "preparation_error",
    "Measurement Error": "measurement_error",
    "Missing Step":      "missing_step",
    "Timing Error":      "timing_error",
    "Temperature Error": "temperature_error",
    "Other":             None,
}


# Used when a single step carries multiple error tags. Structural mistakes
# (missing/reorder) outrank execution mistakes (technique/timing/temperature)
# so the more salient label wins.
PRIORITY: tuple[str, ...] = (
    "missing_step",
    "order_error",
    "preparation_error",
    "measurement_error",
    "technique_error",
    "timing_error",
    "temperature_error",
)


def remap_tag(tag: str) -> Optional[str]:
    """Map one CaptainCook4D error ``tag`` to its VMD class.

    Returns the VMD class string, or ``None`` if the tag is intentionally
    unmapped (currently only ``Other``). Raises ``KeyError`` on an unknown
    tag so data drift is caught loudly.
    """
    if tag not in CC4D_TAG_TO_CLASS:
        raise KeyError(
            f"unknown CaptainCook4D error tag {tag!r}; "
            f"expected one of {sorted(CC4D_TAG_TO_CLASS)}"
        )
    return CC4D_TAG_TO_CLASS[tag]


def step_label(errors: list[dict]) -> str:
    """Reduce a step's error list to a single VMD class.

    ``errors`` is the ``errors`` list from a step_annotation entry, each item
    ``{"tag": <category>, "description": <text>}``. A step with no errors is
    ``no_mistake``. When several mapped tags coexist, ``PRIORITY`` picks one.
    """
    if not errors:
        return "no_mistake"
    mapped = {remap_tag(e["tag"]) for e in errors}
    mapped.discard(None)
    if not mapped:
        return "no_mistake"
    for cls in PRIORITY:
        if cls in mapped:
            return cls
    return sorted(mapped)[0]
