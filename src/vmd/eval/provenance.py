"""Content-addressed manifests for reproducible collection and CPU replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest(config, files):
    payload = {"schema": "vmd_manifest_v1", "config": config,
               "files": {str(Path(p)): file_hash(p) for p in sorted(set(map(Path, files)))}}
    payload["identity"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def guard_collection(path, current, resume=False):
    """Refuse overwrites and resuming data without its matching manifest."""
    path = Path(path)
    metadata = path.with_suffix(path.suffix + ".manifest.json")
    sidecar = path.with_suffix(path.suffix + "l")
    existing_data = path.exists() or sidecar.exists()
    if existing_data and not resume:
        raise ValueError(f"Output exists; use --resume with the same manifest or a new path: {path}")
    if metadata.exists():
        old = json.loads(metadata.read_text())
        if old != current:
            raise ValueError("Run manifest differs: use a new output path")
    elif existing_data:
        raise ValueError("Existing predictions have no manifest; cannot resume under a new protocol")
    else:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(current, indent=2, sort_keys=True))
    if path.exists() and not sidecar.exists():
        raise ValueError("Completed predictions lack a resume sidecar; use a new output path")
