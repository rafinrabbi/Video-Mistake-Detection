from __future__ import annotations
from typing import Dict

from rich.console import Console
from rich.table import Table


def print_results(results: Dict[str, dict], title: str = "Evaluation Results") -> None:
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Mode", style="bold")
    table.add_column("Binary F1")
    table.add_column("Macro F1")
    table.add_column("AUC-ROC")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("ECE")
    table.add_column("Brier")

    for mode_name, mode_results in results.items():
        # run_online_eval returns {"metrics": {...}, "coverage": {...}}
        m = mode_results.get("metrics", mode_results)
        table.add_row(
            mode_name,
            f"{m.get('binary_f1', 0):.3f}",
            f"{m.get('macro_f1', 0):.3f}",
            f"{m.get('auc_roc', 0):.3f}",
            f"{m.get('precision', 0):.3f}",
            f"{m.get('recall', 0):.3f}",
            f"{m.get('ece', 0):.3f}",
            f"{m.get('brier', 0):.3f}",
        )

    console.print(table)
