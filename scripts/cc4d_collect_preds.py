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
from importlib.metadata import version
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--clip_len", type=float, default=4.0)
    ap.add_argument("--stride", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="fixed confidence threshold for the reported metrics")
    ap.add_argument("--gate_mode", default="conf_only", choices=["and", "conf_only"])
    ap.add_argument("--prompt_variant", default="graph",
                    choices=["graph", "nograph", "staticdump", "belief", "belief_nograph", "stepreco"],
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
    ap.add_argument("--tracker", default="pointer", choices=["pointer", "belief"],
                    help="'belief' = DAG-constrained belief filter fed by the model's "
                         "step letter (GraphTrack); 'pointer' = legacy rate-limited pointer")
    ap.add_argument("--oracle_step", action="store_true",
                    help="condition on the GOLD step of each window (upper bound; never a system)")
    ap.add_argument("--filter_json", default=None,
                    help="JSON dict of FilterConfig overrides for the belief tracker")
    ap.add_argument("--recipe_holdout", default=None,
                    help="unseen-recipe split JSON; with --split test evaluates the held-out recipes")
    ap.add_argument("--context_sec", type=float, default=0.0,
                    help="step recogniser look-back in seconds; frames end-context..end at --context_fps")
    ap.add_argument("--context_fps", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="compact belief output needs ~24; keep 512 for legacy prompts")
    ap.add_argument("--evaluation_protocol", choices=["causal_v1", "legacy_v1"], default="causal_v1")
    ap.add_argument("--smooth_k", type=int, default=1)
    ap.add_argument("--min_consecutive", type=int, default=1)
    args = ap.parse_args()
    if args.out and Path(args.out).exists() and not args.resume:
        raise SystemExit("Evaluation output exists; choose a new --out path")

    from vmd.data.captaincook4d import CaptainCook4DDataset
    from vmd.graph.schema import ProceduralGraph

    ds = CaptainCook4DDataset(args.raw_dir, args.split,
                              recipe_holdout=Path(args.recipe_holdout) if args.recipe_holdout else None)
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
        f"prompt={args.prompt_variant} tracker={args.tracker} oracle={args.oracle_step} "
        f"tau={args.tau} gate={args.gate_mode}"
    )

    from vmd.vlm.qwen25vl import Qwen25VL

    from vmd.eval.provenance import manifest, guard_collection
    from vmd.vlm.boolean_scoring import SCORE_VERSION
    config = {k: v for k, v in vars(args).items() if k not in ("resume", "out", "save_preds")}
    config.update(score_version=SCORE_VERSION, recording_ids=[c.clip_id for c in clips],
                  label_policy="historical_other_as_normal_midpoint_v1",
                  graph_source="supplied_graph_files")
    config["packages"] = {name: version(name) for name in ("torch", "transformers", "peft", "numpy", "decord")}
    config["videos"] = {c.clip_id: {"path": str(c.video_path.resolve()),
                                    "bytes": c.video_path.stat().st_size,
                                    "mtime_ns": c.video_path.stat().st_mtime_ns}
                        for c in clips}
    files = list((ROOT / "src/vmd").rglob("*.py")) + list((ROOT / "src/vmd").rglob("*.txt"))
    files += [Path(__file__)] + list(Path(args.graphs_dir).glob("*.json"))
    ann = Path(args.raw_dir) / "annotations"
    for folder, pattern in (("annotation_json", "*.json"), ("annotation_csv", "*.csv"),
                            ("data_splits", "*.json"), ("metadata", "*.csv")):
        files += list((ann / folder).glob(pattern))
    for optional in (args.step_seconds, args.recipe_holdout):
        if optional and Path(optional).exists():
            files.append(Path(optional))
    if args.lora_adapter:
        files += list(Path(args.lora_adapter).glob("*.json"))
        files += list(Path(args.lora_adapter).glob("*.safetensors"))
    # Pin cached base-model snapshot identity without hashing multi-GB shards.
    from transformers.utils.hub import cached_file
    model_config = Path(cached_file("Qwen/Qwen2.5-VL-7B-Instruct", "config.json", local_files_only=True))
    config["base_model_snapshot"] = model_config.parent.name
    files.append(model_config)
    run_manifest = manifest(config, files)
    if args.save_preds:
        guard_collection(Path(args.save_preds), run_manifest, args.resume)

    t0 = time.time()
    vlm = Qwen25VL(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_adapter_dir=args.lora_adapter,
        prompt_variant=args.prompt_variant,
        max_new_tokens=args.max_new_tokens,
        revision=config["base_model_snapshot"],
    )
    print(f"VLM loaded in {time.time() - t0:.1f}s"
          + (f" (LoRA: {args.lora_adapter})" if args.lora_adapter else " (zero-shot)"))

    from vmd.eval.online import run_online_eval

    t1 = time.time()
    result = run_online_eval(
        clips, graphs, vlm,
        clip_len_sec=args.clip_len, stride_sec=args.stride, fps=args.fps,
        fixed_tau=args.tau, gate_mode=args.gate_mode,
        smooth_k=args.smooth_k, min_consecutive=args.min_consecutive,
        evaluation_protocol=args.evaluation_protocol,
        save_preds_path=Path(args.save_preds) if args.save_preds else None,
        log_text=args.log_text,
        resume=args.resume,
        step_seconds_path=Path(args.step_seconds) if args.step_seconds else None,
        tracker=args.tracker,
        oracle_step=args.oracle_step,
        filter_overrides=json.loads(args.filter_json) if args.filter_json else None,
        context_sec=args.context_sec,
        context_fps=args.context_fps,
    )
    dt = time.time() - t1
    print(f"\neval done in {dt:.1f}s ({dt / max(1, n_windows_est):.2f}s/window)")
    _report_peak_memory()

    m = result["metrics"]
    print("\n=== online ===")
    for k in ("binary_f1", "macro_f1", "macro_f1_excl_missing_step", "auc_roc",
              "auc_raw", "average_precision", "precision", "recall", "ece", "brier", "eds", "start_mae",
              "event_precision", "event_recall", "false_alarms_per_min", "median_delay_sec"):
        if k in m and m[k] is not None:
            print(f"  {k}: {m[k]:.4f}")
    print(f"  coverage: {result['coverage']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "config": vars(args),
            "manifest": run_manifest,
            "n_clips": len(clips),
            "results": {"online": result},
        }, indent=2, default=str))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
