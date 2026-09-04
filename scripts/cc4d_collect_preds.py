"""Collect per-window VLM predictions on CaptainCook4D and score them (online).

Runs the causal online evaluator over CaptainCook4D clips using the gold task
graphs in data/graphs. Works for the zero-shot model, a merged LoRA adapter
(--lora_adapter), and the graph-ablation arms (--prompt_variant). The heavy
output is --save_preds: per-window records that scripts/cc4d_decision_eval.py
can re-gate on CPU with any decision config, no GPU needed.

Examples:
  # smoke: 1 clip, coarse stride, measure latency
  python scripts/cc4d_collect_preds.py --limit 1 --stride 8
  # full test pass, LoRA arm, save preds for re-gating
  python scripts/cc4d_collect_preds.py --split test \
      --lora_adapter outputs/lora_cc4d_8class \
      --save_preds outputs/decision_v2/lora_test_preds.json \
      --out outputs/decision_v2/lora_test_eval.json
  # graph-off ablation arm (prompt stripped of graph context; coverage unchanged)
  python scripts/cc4d_collect_preds.py --split test --prompt_variant nograph \
      --lora_adapter outputs/lora_cc4d_8class \
      --save_preds outputs/decision_v2/lora_nograph_test_preds.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _report_peak_memory() -> None:
    """Print peak VRAM and host RSS for this run.

    The GPU here is shared, so 'how much does this actually need' is a question
    that comes up every time a run is scheduled. Measuring it costs nothing and
    turns an estimate into a number we can plan against.
    """
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            resv = torch.cuda.max_memory_reserved() / 2 ** 30
            print(f"peak VRAM: {peak:.2f} GiB allocated, {resv:.2f} GiB reserved")
    except Exception:
        pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        print(f"peak host RSS: {rss:.2f} GiB")
    except Exception:
        pass


def stratified_subset(clips, n):
    """Pick n clips, balancing error vs normal, deterministically.

    Prefer error clips with at least one LOCALIZABLE annotation (start>=0) so
    recall is actually measurable (start=-1 mistakes never mark a window).
    """
    err_loc = [c for c in clips if any(a.start_sec >= 0 for a in c.annotations)]
    norm = [c for c in clips if not c.annotations]
    half = n // 2
    pick = err_loc[:half] + norm[: n - half]
    return pick[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--graphs_dir", default=str(ROOT / "data/graphs"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--clip_len", type=float, default=4.0)
    ap.add_argument("--stride", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="fixed confidence threshold for the reported metrics")
    ap.add_argument("--gate_mode", default="and", choices=["and", "conf_only"])
    ap.add_argument("--prompt_variant", default="graph",
                    choices=["graph", "nograph", "staticdump"],
                    help="'nograph' strips the graph block; 'staticdump' keeps the "
                         "full step list but removes the model's position in it "
                         "(the walked-pointer control). Coverage is unchanged.")
    ap.add_argument("--graph_permute_seed", type=int, default=None,
                    help="derange task_id->graph mapping (wrong-graph control)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_preds", default=None,
                    help="save per-window pred records (for CPU re-gating)")
    ap.add_argument("--step_seconds", default=str(ROOT / "data/step_seconds.json"),
                    help="per-recipe seconds-per-step, estimated on TRAIN only; "
                         "rate-limits the step pointer so it tracks the person's "
                         "progress instead of the window counter")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: reuse the per-clip records "
                         "already in the <save_preds>l sidecar and skip those clips")
    ap.add_argument("--log_text", action="store_true",
                    help="also save the model's rationale, observed action/objects, "
                         "uncertainty reason and verbalized confidence per window "
                         "(for qualitative figures; multiplies preds file size)")
    ap.add_argument("--lora_adapter", default=None,
                    help="load + merge a LoRA adapter dir (Contribution A fine-tuned)")
    args = ap.parse_args()

    from vmd.data.captaincook4d import CaptainCook4DDataset
    from vmd.graph.schema import ProceduralGraph

    ds = CaptainCook4DDataset(args.raw_dir, args.split)
    clips = ds.clips
    if args.stratify and args.limit:
        clips = stratified_subset(clips, args.limit)
    elif args.limit:
        clips = clips[: args.limit]

    graphs = {
        p.stem: ProceduralGraph.model_validate_json(p.read_text())
        for p in Path(args.graphs_dir).glob("*.json")
    }

    if args.graph_permute_seed is not None:
        # Wrong-graph control: deterministically map every evaluated task to a
        # DIFFERENT task's graph (fixpoint-free permutation over the tasks that
        # appear in this split). Coverage is unchanged: every task still has a
        # graph; only the graph CONTENT is wrong.
        import random
        task_ids = sorted({c.task_id for c in clips if c.task_id in graphs})
        if len(task_ids) < 2:
            raise SystemExit("need >=2 tasks for a wrong-graph derangement")
        rng = random.Random(args.graph_permute_seed)
        shifted = task_ids[:]
        rng.shuffle(shifted)
        if any(a == b for a, b in zip(task_ids, shifted)):
            shifted = task_ids[1:] + task_ids[:1]  # rotation: fixpoint-free
        graphs = {**graphs, **{t: graphs[s] for t, s in zip(task_ids, shifted)}}
        print(f"wrong-graph control: deranged {len(task_ids)} tasks "
              f"(seed {args.graph_permute_seed})")

    covered = sum(1 for c in clips if c.task_id in graphs)
    n_windows_est = sum(
        max(1, int((c.duration_sec - args.clip_len) // args.stride) + 1) for c in clips
    )
    print(
        f"clips={len(clips)} (graph-covered={covered}) "
        f"est_windows={n_windows_est} stride={args.stride}s "
        f"prompt={args.prompt_variant} tau={args.tau} gate={args.gate_mode}"
    )

    from vmd.vlm.qwen25vl import Qwen25VL

    t0 = time.time()
    vlm = Qwen25VL(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_adapter_dir=args.lora_adapter,
        prompt_variant=args.prompt_variant,
    )
    print(f"VLM loaded in {time.time() - t0:.1f}s"
          + (f" (LoRA: {args.lora_adapter})" if args.lora_adapter else " (zero-shot)"))

    from vmd.eval.online import run_online_eval

    t1 = time.time()
    result = run_online_eval(
        clips, graphs, vlm,
        clip_len_sec=args.clip_len, stride_sec=args.stride, fps=args.fps,
        fixed_tau=args.tau, gate_mode=args.gate_mode,
        save_preds_path=Path(args.save_preds) if args.save_preds else None,
        log_text=args.log_text,
        resume=args.resume,
        step_seconds_path=Path(args.step_seconds) if args.step_seconds else None,
    )
    dt = time.time() - t1
    print(f"\neval done in {dt:.1f}s ({dt / max(1, n_windows_est):.2f}s/window)")
    _report_peak_memory()

    m = result["metrics"]
    print("\n=== online ===")
    for k in ("binary_f1", "macro_f1", "macro_f1_excl_missing_step", "auc_roc",
              "auc_raw", "precision", "recall", "ece", "brier", "eds", "start_mae"):
        if k in m:
            print(f"  {k}: {m[k]:.4f}")
    print(f"  coverage: {result['coverage']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "config": vars(args),
            "n_clips": len(clips),
            "results": {"online": result},
        }, indent=2, default=str))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
