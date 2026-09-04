"""Generate explanations on the test set and score them (§4.1).

Loads:
  - Stage 1 backbone weights (Q-Formers, hand encoder, mistake head)
  - Stage 2 llm_proj weights (`*.pt` saved by train_stage2)
  - Optional Stage 3 LoRA adapter dir (`*_lora/`)

For each test clip:
  1. Run frozen visual encoder → Q-Former → llm_proj → soft-prefix
  2. Generate explanation via `llm.generate(inputs_embeds=...)`
  3. Compare to ground-truth explanation.txt

Computes BLEU / ROUGE-L / CIDEr over all clips with non-empty references.
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

from src.caption_metrics import all_caption_metrics
from src.llm import build_llm, generate_explanation
from src.model import MistSense, ModelConfig
from src.utils import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--override", default=str(ROOT / "configs/egoops.yaml"))
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_annot", required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True, help="dict with llm_proj state")
    ap.add_argument("--lora_dir", default=None, help="Stage 3 LoRA adapter dir")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.base, args.override)
    mcfg = ModelConfig(**{k: cfg.model[k] for k in cfg.model if k in ModelConfig.__dataclass_fields__})
    model = MistSense(mcfg).to(args.device)
    sd1 = torch.load(args.stage1_ckpt, map_location=args.device)
    sd1 = {k: v for k, v in sd1.items() if not k.startswith("llm_proj.")}
    model.load_state_dict(sd1, strict=False)
    proj_state = torch.load(args.stage2_ckpt, map_location=args.device)["llm_proj"]
    model.llm_proj.load_state_dict(proj_state)
    model.eval()

    llm, tokenizer, _ = build_llm(
        pretrained=cfg.model.get("llm_pretrained", "Salesforce/blip2-opt-2.7b"),
        from_blip2=bool(cfg.model.get("llm_from_blip2", True)),
        device=args.device,
    )
    if args.lora_dir:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, args.lora_dir)
    llm.eval()

    entries = json.loads(Path(args.test_annot).read_text())
    data_root = Path(args.data_root)
    preds, refs, ids = [], [], []
    for e in entries:
        cid = e["clip_id"]
        d = data_root / cid
        if not d.exists():
            continue
        ref = (d / "explanation.txt").read_text().strip()
        if not ref:
            continue  # skip no_mistake clips for caption metrics
        visual = torch.from_numpy(np.load(d / "visual.npy")).float()
        # Pick the window with highest mistake probability per Stage-1 gate
        T = visual.size(0)
        t_s = mcfg.segment_length
        if T < t_s:
            visual = torch.nn.functional.pad(visual, (0, 0, 0, t_s - T))
            T = t_s
        labels = np.load(d / "labels.npy")
        # Take the densest mistake window if any, else center crop
        if labels.sum() > 0:
            # window centered on the densest mistake region
            cumsum = np.concatenate([[0], np.cumsum(labels)])
            scores = cumsum[t_s:] - cumsum[:-t_s]
            start = int(np.argmax(scores))
        else:
            start = max(0, (T - t_s) // 2)
        v = visual[start:start + t_s].unsqueeze(0).to(args.device)
        with torch.no_grad():
            f_rgb = model.encode_video(v)
            prefix = model.project_to_llm(f_rgb).to(torch.float16)
            txt = generate_explanation(llm, tokenizer, prefix, max_new_tokens=args.max_new_tokens)[0]
        preds.append(txt.strip())
        refs.append(ref)
        ids.append(cid)
        print(f"{cid}: pred={txt[:80]!r}\n        ref={ref[:80]!r}")

    if not preds:
        print("no test clips with non-empty references; skipping metrics")
        return
    m = all_caption_metrics(preds, refs)
    print("\n=== caption metrics ===")
    for k, v in m.items():
        print(f"  {k}: {v}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "metrics": m,
            "per_clip": [{"clip_id": c, "pred": p, "ref": r} for c, p, r in zip(ids, preds, refs)],
        }, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
