"""Assemble the online 8-class CC4D ablation table from decision-row JSONs.

Consumes a MANIFEST: an ordered list of {name, row (path), [bootstrap (path)],
[note]} entries, where each `row` file is the uniform output of
scripts/cc4d_decision_eval.py / cc4d_decision_sweep.py ({"config": .., \
"metrics": ..}) — every table row, GPU-collected or CPU-re-gated, goes through
the same decision-layer code path. Optional `bootstrap` files (from
cc4d_bootstrap_compare.py) attach 95% CIs.

  python scripts/cc4d_build_ablation.py \
      --manifest outputs/decision_v2/ablation_manifest.json \
      --out outputs/decision_v2/ablation.json \
      --md_out outputs/decision_v2/ablation.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmd.eval.metrics import MISTAKE_CLASSES  # noqa: E402

AGG_COLS = ["binary_f1", "macro_f1", "macro_f1_excl_missing_step",
            "auc_raw", "precision", "recall", "eds", "start_mae"]
CI_COLS = {"binary_f1", "macro_f1", "auc_raw", "eds"}


def _row(entry: dict) -> dict:
    data = json.loads(Path(entry["row"]).read_text())
    m = data["metrics"]
    pcf = m.get("per_class_f1", {})
    if pcf:
        assert set(pcf.keys()) == set(MISTAKE_CLASSES), (
            f"per_class_f1 keys {sorted(pcf)} != MISTAKE_CLASSES "
            f"{sorted(MISTAKE_CLASSES)} — class taxonomy drift; refusing to "
            "build a silently-misaligned table."
        )
    row = {"row": entry["name"], "config": data.get("config"),
           "note": entry.get("note")}
    for c in AGG_COLS:
        row[c] = m.get(c)
    for c in MISTAKE_CLASSES:
        row[f"f1_{c}"] = pcf.get(c)
    row["support"] = m.get("support")
    row["n_loc_events"] = m.get("n_loc_events")
    row["n_loc_detected"] = m.get("n_loc_detected")
    row["n_windows"] = m.get("n_windows")
    row["n_clips"] = m.get("n_clips")

    if entry.get("bootstrap"):
        boot = json.loads(Path(entry["bootstrap"]).read_text())
        row["ci95"] = boot.get("ci95")
    return row


def _fmt(v, ci=None) -> str:
    if v is None:
        return "—"
    s = f"{v:.3f}" if isinstance(v, float) else str(v)
    if ci:
        s += f" [{ci[0]:.3f},{ci[1]:.3f}]"
    return s


def _md_table(rows: list[dict]) -> str:
    cols = ["row"] + AGG_COLS
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        cells = [r["row"]]
        for c in cols[1:]:
            ci = (r.get("ci95") or {}).get(c) if c in CI_COLS else None
            cells.append(_fmt(r.get(c), ci))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_per_class(rows: list[dict]) -> str:
    cols = ["row"] + [f"f1_{c}" for c in MISTAKE_CLASSES]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        if all(r.get(f"f1_{c}") is None for c in MISTAKE_CLASSES):
            continue
        lines.append("| " + " | ".join(
            [r["row"]] + [_fmt(r.get(f"f1_{c}")) for c in MISTAKE_CLASSES]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw/captaincook4d"))
    ap.add_argument("--out", default=str(ROOT / "outputs/decision_v2/ablation.json"))
    ap.add_argument("--md_out", default=str(ROOT / "outputs/decision_v2/ablation.md"))
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for entry in manifest["rows"]:
        if not Path(entry["row"]).exists():
            print(f"SKIP (row file missing yet): {entry['name']} <- {entry['row']}")
            continue
        # Some rows are deliberately not window-matched (R8 runs the baseline's
        # native per-frame protocol) and carry their own bespoke shape. They
        # belong in the footnotes, not in a table whose every other row shares
        # one window grid; skip rather than crash or, worse, coerce.
        if "metrics" not in json.loads(Path(entry["row"]).read_text()):
            print(f"SKIP (not a decision-layer row): {entry['name']}")
            continue
        rows.append(_row(entry))

    # Honest coverage: re-load dataset (cheap, no GPU) for nonlocalizable/untyped.
    from vmd.data.captaincook4d import CaptainCook4DDataset
    ds = CaptainCook4DDataset(args.raw_dir, "test")
    coverage_dataset = {
        "test_clips": len(ds.clips),
        "nonlocalizable_segments": ds.nonlocalizable,
        "untyped_dropped_segments": ds.untyped_dropped,
    }

    footnotes = manifest.get("footnotes", []) + [
        "Protocol: online causal, clip_len 4s, stride 4s, fps 1; trailing <stride s per clip unsampled (floor tiling).",
        f"{coverage_dataset['nonlocalizable_segments']} nonlocalizable (start=-1, mostly Missing-Step) segments mark NO window and are unrecallable by windowed metrics; "
        f"{coverage_dataset['untyped_dropped_segments']} 'Other'-only step(s) dropped from typed metrics. missing_step is structurally unreachable -> macro_f1_excl_missing_step.",
        "CIs: clip-level bootstrap, n=1000, seed=0, percentile 95%. Windows within a clip are correlated; clips are the resampling unit.",
        "EDS must be read jointly with precision: a system that fires early and often scores high EDS regardless of accuracy.",
        "Localization uses min_consecutive persistence from each row's decision config; binary F1 gates on smoothed score>0 — different decision rules by design.",
        "auc_raw ranks by the raw (ungated) per-window confidence — threshold-free, comparable across rows.",
    ]

    ablation = {"dataset_coverage": coverage_dataset, "rows": rows,
                "footnotes": footnotes}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ablation, indent=2, default=str))

    md = ["# CC4D 8-class ablation — online causal, all test clips", "",
          f"Test clips: {coverage_dataset['test_clips']} | "
          f"nonlocalizable: {coverage_dataset['nonlocalizable_segments']} | "
          f"untyped dropped: {coverage_dataset['untyped_dropped_segments']}", "",
          _md_table(rows), "",
          "### Per-class F1 (appendix)", "", _md_per_class(rows), "",
          "### Footnotes", ""]
    md += [f"{i+1}. {f}" for i, f in enumerate(footnotes)]
    md_text = "\n".join(md)
    if args.md_out:
        Path(args.md_out).write_text(md_text)
    print(md_text)
    print(f"\nsaved -> {args.out}\nsaved -> {args.md_out}")


if __name__ == "__main__":
    main()
