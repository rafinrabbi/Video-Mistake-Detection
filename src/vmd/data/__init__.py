"""Dataset loaders. CaptainCook4D is the sole benchmark; all loaders expose
``.clips`` as a list of ``EgoOopsClip`` (legacy dataclass name) so eval and
training code paths stay dataset-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from vmd.data.egoops import EgoOopsClip, MistakeAnnotation

DATASETS = {
    "captaincook4d": "captaincook4d",
    "cc4d": "captaincook4d",
}


def get_dataset(name: str, raw_dir: Union[str, Path], split: str = "train", **kwargs):
    """Factory: return a dataset loader by name; all expose ``.clips``."""
    key = DATASETS.get(name.lower())
    if key == "captaincook4d":
        from vmd.data.captaincook4d import CaptainCook4DDataset
        return CaptainCook4DDataset(Path(raw_dir), split, **kwargs)
    raise ValueError(f"unknown dataset {name!r}; choices: {sorted(DATASETS)}")


__all__ = [
    "EgoOopsClip",
    "MistakeAnnotation",
    "get_dataset",
    "DATASETS",
]
