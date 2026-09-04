"""Per-clip feature extraction for MistSense on EgoOops.

For each video in `videos_dir`:
  - sample frames at 2 fps (§4.6)
  - run BLIP-2 ViT + Q-Former; mean-pool 32 queries -> (768,) per frame
  - run Mediapipe Hands; pack (x,y,z) for 21 joints, 2 hands per frame
  - build binary mistake labels per frame from annotation JSON

Saves:
  out_dir/<clip_id>/visual.npy   (T, 768) float32
  out_dir/<clip_id>/hand.npy     (3, T, 21, 2) float32
  out_dir/<clip_id>/labels.npy   (T,) float32
  out_dir/<clip_id>/explanation.txt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch  # import torch first so its CUDA driver init wins over TF/mediapipe
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import mediapipe as mp  # imports tensorflow lite + tries CUDA stub — must come after torch

FPS = 2  # §4.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISUAL_BATCH = 32


def build_blip2():
    """[DEVIATION] Paper says "ViT-L" + BLIP-2 Q-Former. No public BLIP-2 has
    ViT-L (opt-2.7b uses ViT-g, far slower). We use CLIP-ViT-L/14 (1024-dim
    CLS embedding) which matches the paper's stated visual encoder size
    exactly and is ~10x faster. The base BLIP-2 Q-Former step is collapsed
    into the trainable Video Q-Former. See REPRODUCTION_NOTES.md."""
    model_id = "openai/clip-vit-large-patch14"
    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPVisionModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()
    return proc, model


@torch.no_grad()
def extract_visual(model, proc, frames: list[np.ndarray]) -> np.ndarray:
    """frames: list of HxWx3 uint8 BGR. Returns (T, 1024) — CLIP ViT-L CLS."""
    feats = []
    for i in range(0, len(frames), VISUAL_BATCH):
        batch = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[i : i + VISUAL_BATCH]]
        inputs = proc(images=batch, return_tensors="pt").to(DEVICE, torch.float16)
        out = model(pixel_values=inputs["pixel_values"])
        # pooler_output: (B, 1024) — post-LN CLS
        feats.append(out.pooler_output.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_mp_hands(model_path: str = None):
    """Mediapipe Tasks API (legacy `mp.solutions.hands` removed in py 3.13)."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    if model_path is None:
        model_path = str(Path(__file__).parent / "hand_landmarker.task")
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_hand(hands, frames: list[np.ndarray]) -> np.ndarray:
    """Returns (3, T, 21, 2). Missing detections filled with zeros."""
    T = len(frames)
    out = np.zeros((3, T, 21, 2), dtype=np.float32)
    for t, f in enumerate(frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = hands.detect(mp_image)
        if not res.hand_landmarks:
            continue
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            label = handed[0].category_name  # 'Left' or 'Right'
            slot = 0 if label == "Left" else 1
            for j, lm in enumerate(lms):
                out[0, t, j, slot] = lm.x
                out[1, t, j, slot] = lm.y
                out[2, t, j, slot] = lm.z
    return out


def sample_frames(video_path: str, fps: int = FPS) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    frames = []
    i = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def build_labels(meta: dict, T: int, fps: int = FPS) -> np.ndarray:
    """Binary per-frame: 1 if any mistake span covers frame time t/fps."""
    labels = np.zeros(T, dtype=np.float32)
    for m in meta.get("mistakes", []):
        s = int(math.floor(m["start_sec"] * fps))
        e = int(math.ceil(m["end_sec"] * fps))
        s = max(0, s)
        e = min(T, e)
        labels[s:e] = 1.0
    return labels


def build_explanation(meta: dict) -> str:
    """Concatenate mistake descriptions; empty if no mistakes (§3.2 multi-mistake -> concat)."""
    parts = [m.get("description", m.get("mistake_class", "")) for m in meta.get("mistakes", [])]
    return "; ".join(p for p in parts if p)


def _resolve_video(videos_dir: Path, cid: str) -> Path:
    """Find <cid>.mp4, else <cid>.* / <cid>_* (e.g. CC4D '<rid>_360p.mp4').

    Globs '<cid>.*' then '<cid>_*' (NOT bare '<cid>*') so prefix-ambiguous ids
    like 1_7 vs 1_70 don't cross-match (the '_' delimiter separates them).
    """
    exact = videos_dir / f"{cid}.mp4"
    if exact.exists():
        return exact
    matches = sorted(videos_dir.glob(f"{cid}.*")) or sorted(videos_dir.glob(f"{cid}_*"))
    return matches[0] if matches else exact


def process_clip(clip_meta: dict, videos_dir: Path, out_dir: Path, proc, vmodel, hands):
    cid = clip_meta["clip_id"]
    vp = _resolve_video(videos_dir, cid)
    if not vp.exists():
        print(f"skip {cid}: video missing")
        return
    out = out_dir / cid
    out.mkdir(parents=True, exist_ok=True)
    if (out / "visual.npy").exists() and (out / "hand.npy").exists() and (out / "labels.npy").exists():
        print(f"skip {cid}: already extracted")
        return
    frames = sample_frames(str(vp))
    T = len(frames)
    visual = extract_visual(vmodel, proc, frames)
    hand = extract_hand(hands, frames)
    labels = build_labels(clip_meta, T)
    expl = build_explanation(clip_meta)
    np.save(out / "visual.npy", visual)
    np.save(out / "hand.npy", hand)
    np.save(out / "labels.npy", labels)
    (out / "explanation.txt").write_text(expl)
    print(f"{cid}: T={T} visual={visual.shape} hand={hand.shape} mistakes={int(labels.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--annot_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    proc, vmodel = build_blip2()
    hands = build_mp_hands()

    annot_dir = Path(args.annot_dir)
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)

    all_meta: list[dict] = []
    for split in ["train", "val", "test"]:
        p = annot_dir / f"{split}.json"
        if p.exists():
            all_meta.extend(json.loads(p.read_text()))

    for clip_meta in all_meta:
        process_clip(clip_meta, videos_dir, out_dir, proc, vmodel, hands)


if __name__ == "__main__":
    main()
