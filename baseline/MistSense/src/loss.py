"""MistSense — losses.

§3.3 Learning Objectives.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """§3.3 — Stage 1: weighted binary cross-entropy on mistake logits.

    Paper formulation (equation rendered as image, prose):
        L_det = - (1/N) Σ_i [ w_1 * y_i * log(ŷ_i) + w_0 * (1 - y_i) * log(1 - ŷ_i) ]
    where y_i = 1 denotes mistake, y_i = 0 correct action, ŷ_i is the
    sigmoid-passed logit. w_0 / w_1 are class weights (§4.6:
    Epic-tent w_0=w_1=1, HoloAssist w_0=1, w_1=4).
    """

    def __init__(self, w0: float = 1.0, w1: float = 1.0):
        super().__init__()
        self.w0 = w0
        self.w1 = w1

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, t_q) raw, target: (B, t_q) in {0,1}
        """
        probs = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
        loss_pos = -self.w1 * target * torch.log(probs)
        loss_neg = -self.w0 * (1.0 - target) * torch.log(1.0 - probs)
        return (loss_pos + loss_neg).mean()


def explanation_ce_loss(llm_logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """§3.3 — Stage 2/3: standard CE on LLM next-token logits.

    llm_logits: (B, T, V), labels: (B, T) with -100 to mask non-target tokens.
    """
    return F.cross_entropy(
        llm_logits.reshape(-1, llm_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
