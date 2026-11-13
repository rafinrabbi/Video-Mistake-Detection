"""Render the frozen matched-graph PR and paired AP comparison as PNG/PDF."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    summary = json.loads((run / "summary.json").read_text())
    curves = {name: ([], []) for name in ("content", "dag", "linear", "uniform")}
    with (run / "pr_curves.csv").open() as f:
        for row in csv.DictReader(f):
            p, r = curves[row["arm"]]
            p.append(float(row["precision"]))
            r.append(float(row["recall"]))
    labels = dict(content="Content only", dag="DAG + content",
                  linear="Linear + content", uniform="No-graph temporal + content")
    colors = dict(content="#666666", dag="#0072B2", linear="#D55E00", uniform="#009E73")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    for name, (p, r) in curves.items():
        score = summary["arms"][name]["window_f1"]["average_precision"]
        axes[0].step(r, p, where="post", lw=1.6, color=colors[name],
                     label=f"{labels[name]} (AP {score:.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
                title="Cached test scores: 109 recordings")
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    comparisons = summary["bootstrap"]["comparisons"]
    for i, other in enumerate(("linear", "uniform", "content")):
        d = comparisons[f"dag_minus_{other}"]["average_precision"]
        lo, hi = d["ci95"]
        axes[1].errorbar(d["point"], i, xerr=[[d["point"] - lo], [hi - d["point"]]],
                         fmt="o", color=colors[other], capsize=4)
    axes[1].axvline(0, color="black", lw=.8, linestyle="--")
    axes[1].set(yticks=np.arange(3), yticklabels=["DAG − Linear", "DAG − No graph", "DAG − Content"],
                xlabel="Paired difference in average precision",
                title="95% recording-bootstrap intervals", ylim=(-.5, 2.5))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.suptitle("Matched topology replay · exploratory · shared historical content scores", fontsize=10)
    for suffix in ("png", "pdf"):
        path = run / f"matched_graph_comparison.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=220)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
