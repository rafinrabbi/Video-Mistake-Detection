from __future__ import annotations
import torch
import torch.nn.functional as F


def mistake_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def calibration_loss(confidences: torch.Tensor, correctness: torch.Tensor) -> torch.Tensor:
    """NLL-based calibration: penalize overconfident wrong predictions."""
    correctness_f = correctness.float()
    # Binary NLL: -[y·log(p) + (1-y)·log(1-p)]
    eps = 1e-7
    p = confidences.clamp(eps, 1 - eps)
    return -(correctness_f * torch.log(p) + (1 - correctness_f) * torch.log(1 - p)).mean()


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    ce_weight: float = 1.0,
    calib_weight: float = 0.3,
) -> torch.Tensor:
    return ce_weight * mistake_ce_loss(logits, labels) + calib_weight * calibration_loss(confidences, correctness)
