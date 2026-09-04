"""Shared utilities for MistSense reproduction.

Paper: Patsch et al., MistSense (ICCV 2025)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Flattened access to nested YAML config."""

    model: dict
    training: dict
    data: dict
    eval: dict

    @classmethod
    def load(cls, base_path: str | Path, override_path: str | Path | None = None) -> "Config":
        """Load base config, optionally override with dataset-specific YAML."""
        with open(base_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        if override_path is not None:
            with open(override_path) as f:
                ov = yaml.safe_load(f) or {}
            for k, v in ov.items():
                cfg.setdefault(k, {}).update(v)
        return cls(model=cfg["model"], training=cfg["training"], data=cfg["data"], eval=cfg["eval"])


def count_trainable(module) -> int:
    """Return number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def freeze(module) -> None:
    """In-place freeze: set requires_grad=False on all params."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module) -> None:
    for p in module.parameters():
        p.requires_grad = True
