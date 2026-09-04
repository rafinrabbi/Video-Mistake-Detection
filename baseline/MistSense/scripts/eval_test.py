"""Test-set evaluation for a Stage-1 MistSense checkpoint.

Loads a state_dict saved by `train.train_stage1`, runs over EgoOops test
clips, and reports detection metrics (§4.1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import MistakeClipDataset
from src.evaluate import detection_metrics
from src.model import MistSense, ModelConfig
from src.utils import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--override", default=str(ROOT / "configs/egoops.yaml"))
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_annot", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.load(args.base, args.override)
    mcfg = ModelConfig(**{k: cfg.model[k] for k in cfg.model if k in ModelConfig.__dataclass_fields__})
    model = MistSense(mcfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    miss, unexp = model.load_state_dict(state, strict=False)
    if miss or unexp:
        print(f"WARN: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    ds = MistakeClipDataset(
        data_root=args.data_root,
        segment_length=mcfg.segment_length,
        num_queries=mcfg.num_queries,
        annot_file=args.test_annot,
    )
    loader = DataLoader(ds, batch_size=cfg.data["batch_size"], shuffle=False, num_workers=cfg.data["num_workers"])
    print(f"test windows: {len(ds)}  from clips: {len(ds.clip_ids)}")

    logits_all, target_all = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["visual"].to(args.device), batch["hand"].to(args.device))
            logits_all.append(out["logits"].flatten().cpu())
            target_all.append(batch["target"].flatten())
    logits = torch.cat(logits_all)
    target = torch.cat(target_all)
    print(f"positives in test: {int(target.sum().item())}/{target.numel()}")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = detection_metrics(logits, target, threshold=thresh)
        print(f"τ={thresh:.2f} | F1={m['f1_overall']:.4f} | "
              f"f1_mistake={m['f1_mistake']:.4f} "
              f"(P={m['precision_mistake']:.3f} R={m['recall_mistake']:.3f}) | "
              f"f1_correct={m['f1_correct']:.4f} "
              f"(P={m['precision_correct']:.3f} R={m['recall_correct']:.3f})")


if __name__ == "__main__":
    main()
