"""Online causal sliding-window evaluator for MistSense (§3.2, §4.1).

For each test clip, slide a window of length t_s causally over frames.
At time t (>= t_s-1) the model sees [t-t_s+1, t] and predicts a per-segment
sigmoid; we attribute the prediction to frame t (the rightmost frame).
Frames t < t_s-1 are predicted with right-padded windows so we get a
prediction for every frame.

Aggregates per-frame predictions across all test clips and reports F1
metrics over τ sweep (matches Stage 1 protocol but per-frame instead of
per-window).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import detection_metrics
from src.model import MistSense, ModelConfig
from src.utils import Config


def load_clip(data_root: Path, clip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = data_root / clip_id
    return (
        np.load(d / "visual.npy"),
        np.load(d / "hand.npy"),
        np.load(d / "labels.npy"),
    )


def online_predict_clip(
    model: MistSense,
    visual: np.ndarray,
    hand: np.ndarray,
    t_s: int,
    device: str,
    batch: int = 32,
) -> np.ndarray:
    """Returns (T,) sigmoid probabilities, one per frame (causal)."""
    T, d_v = visual.shape
    c, T_h, V, M = hand.shape
    assert T == T_h, "visual/hand T mismatch"
    visual_t = torch.from_numpy(visual).float()
    hand_t = torch.from_numpy(hand).float()

    starts = np.arange(0, T, dtype=np.int64)
    ends = np.minimum(starts + t_s, T)  # right pad if needed
    probs = np.zeros(T, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, T, batch):
            j = min(i + batch, T)
            v_batch, h_batch = [], []
            for s, e in zip(starts[i:j], ends[i:j]):
                # window: rightmost frame = s+t_s-1 if causal w/ history.
                # For causal online, anchor at frame t and use history
                # [t-t_s+1 .. t]. Reformulate:
                t = int(s)
                lo = max(0, t - t_s + 1)
                hi = t + 1
                v = visual_t[lo:hi]
                h = hand_t[:, lo:hi]
                pad = t_s - v.size(0)
                if pad > 0:
                    # left-pad with zeros (no future leakage)
                    v = torch.cat([torch.zeros(pad, v.size(1)), v], dim=0)
                    h = torch.cat([torch.zeros(c, pad, V, M), h], dim=1)
                v_batch.append(v)
                h_batch.append(h)
            vb = torch.stack(v_batch).to(device)
            hb = torch.stack(h_batch).to(device)
            out = model(vb, hb)
            # gate = max sigmoid over t_q queries -> per-segment mistake prob
            probs[i:j] = out["gate"].cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--override", default=str(ROOT / "configs/egoops.yaml"))
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_annot", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump_probs", default=None,
                    help="dir to write per-clip per-frame probs/labels .npy "
                        "(for re-tiling onto the VLM window grid)")
    args = ap.parse_args()

    cfg = Config.load(args.base, args.override)
    mcfg = ModelConfig(**{k: cfg.model[k] for k in cfg.model if k in ModelConfig.__dataclass_fields__})
    model = MistSense(mcfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    state = {k: v for k, v in state.items() if not k.startswith("llm_proj.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    entries = json.loads(Path(args.test_annot).read_text())
    data_root = Path(args.data_root)
    all_probs, all_labels = [], []
    for e in entries:
        cid = e["clip_id"]
        if not (data_root / cid).exists():
            continue
        visual, hand, labels = load_clip(data_root, cid)
        probs = online_predict_clip(model, visual, hand, mcfg.segment_length, args.device)
        all_probs.append(probs)
        all_labels.append(labels)
        if args.dump_probs:
            dump_dir = Path(args.dump_probs) / cid
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(dump_dir / "probs.npy", probs)
            np.save(dump_dir / "labels.npy", labels)
        print(f"{cid}: T={len(labels)} positives={int(labels.sum())} mean_prob={probs.mean():.3f}")
    probs = torch.from_numpy(np.concatenate(all_probs))
    labels = torch.from_numpy(np.concatenate(all_labels))
    # `detection_metrics` expects logits; pass logits = logit(probs) so
    # internal sigmoid recovers probs. Equivalent: bypass sigmoid by passing
    # (probs - 0) huge values. Simpler: monkey-rewrap.
    logits = torch.log(probs.clamp(1e-7, 1 - 1e-7) / (1 - probs.clamp(1e-7, 1 - 1e-7)))
    print(f"\ntotal frames: {labels.numel()} positives: {int(labels.sum())}")
    rows = []
    for tau in [round(0.05 * i, 2) for i in range(1, 20)]:
        m = detection_metrics(logits, labels, threshold=tau)
        rows.append({"tau": tau, **m})
        print(f"τ={tau:.2f} | F1={m['f1_overall']:.4f} | "
              f"f1_mistake={m['f1_mistake']:.4f} "
              f"(P={m['precision_mistake']:.3f} R={m['recall_mistake']:.3f}) | "
              f"f1_correct={m['f1_correct']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
