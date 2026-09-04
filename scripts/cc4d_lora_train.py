"""Contribution A: LoRA fine-tune Qwen2.5-VL-7B graph-conditioned on CaptainCook4D.

Balanced 50/50 window subset (~10k), stride 4s. ~6-7h for 2 epochs on one
96GB GPU. Adapters -> outputs/lora_cc4d/.

  python scripts/cc4d_lora_train.py
  python scripts/cc4d_lora_train.py --epochs 3 --max_windows 14000 --no_wandb
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.train.lora_train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--output_dir", default=str(ROOT / "outputs/lora_cc4d"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--checkpoint_every", type=int, default=50,
                    help="save optimiser+scheduler+adapter every N optimiser steps "
                         "(0 disables); resume granularity on a shared GPU")
    ap.add_argument("--resume", action="store_true",
                    help="continue from output_dir/training_state.pt if present")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_len", type=float, default=4.0)
    ap.add_argument("--stride", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--neg_per_pos", type=float, default=1.0)
    ap.add_argument("--max_windows", type=int, default=10000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--prompt_variant", default="graph", choices=["graph", "nograph"],
                    help="'nograph' trains with the graph block stripped from the "
                         "prompt (graph ablation arm)")
    args = ap.parse_args()

    train(
        model_id=args.model_id,
        dataset_name="captaincook4d",
        raw_dir=args.raw_dir,
        graphs_dir=args.graphs_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        grad_accum=args.grad_accum,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        seed=args.seed,
        clip_len_sec=args.clip_len,
        stride_sec=args.stride,
        fps=args.fps,
        neg_per_pos=args.neg_per_pos,
        max_windows=args.max_windows,
        num_workers=args.num_workers,
        wandb_project="vmd-cc4d-lora",
        use_wandb=not args.no_wandb,
        prompt_variant=args.prompt_variant,
    )


if __name__ == "__main__":
    main()
