"""MistSense — evaluation metrics (§4.1).

Mistake detection: F1, precision, recall per class + overall F1.
HoloAssist test labels are not public — submit predictions to Codabench.

Caption metrics (BLEU / ROUGEL / CIDEr) are out of scope for the minimal
reproduction; the official `pycocoevalcap` package implements all three.
"""
from __future__ import annotations

from typing import Sequence

import torch


def detection_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Per §4.1: precision, recall, F1 for correct and mistake classes.

    logits, target: (N,) flattened across batch/queries.
    """
    pred = (torch.sigmoid(logits) >= threshold).float()
    tgt = target.float()
    eps = 1e-12

    # Mistake class (label = 1)
    tp_m = ((pred == 1) & (tgt == 1)).sum().item()
    fp_m = ((pred == 1) & (tgt == 0)).sum().item()
    fn_m = ((pred == 0) & (tgt == 1)).sum().item()
    prec_m = tp_m / (tp_m + fp_m + eps)
    rec_m = tp_m / (tp_m + fn_m + eps)
    f1_m = 2 * prec_m * rec_m / (prec_m + rec_m + eps)

    # Correct class (label = 0)
    tp_c = ((pred == 0) & (tgt == 0)).sum().item()
    fp_c = ((pred == 0) & (tgt == 1)).sum().item()
    fn_c = ((pred == 1) & (tgt == 0)).sum().item()
    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c = tp_c / (tp_c + fn_c + eps)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)

    return {
        "f1_overall": 0.5 * (f1_m + f1_c),
        "f1_mistake": f1_m,
        "precision_mistake": prec_m,
        "recall_mistake": rec_m,
        "f1_correct": f1_c,
        "precision_correct": prec_c,
        "recall_correct": rec_c,
    }
