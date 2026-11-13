"""Positive-class scoring at a shared generated JSON-field prefix.

This is conditional on the generated prefix and on the boolean candidate set,
not a calibrated probability of a real-world error. Unsupported tokenizations
return no score; generation confidence is never substituted for class evidence.
"""
from __future__ import annotations

import re

SCORE_VERSION = "boolean_pair_v1"
_FIELD = re.compile(r'"deviation"\s*:\s*$')


def deviation_probability(tokenizer, gen_ids, scores):
    """Normalize true/false logits at the SAME position, independent of the
    generated decision. Include bare and space-prefixed spellings symmetrically.
    All candidates must be single tokens with exact decoding (Qwen contract).
    """
    import torch

    ids = list(gen_ids)
    candidates = {}
    for label in ("true", "false"):
        values = set()
        for spelling in (label, " " + label):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) != 1 or tokenizer.decode(encoded) != spelling:
                return None
            values.add(encoded[0])
        candidates[label] = sorted(values)
    for k, token in enumerate(ids):
        if k >= len(scores):
            break
        if not _FIELD.search(tokenizer.decode(ids[:k])):
            continue
        if tokenizer.decode([token]).strip() not in ("true", "false"):
            continue
        logits = scores[k][0].float()
        masses = torch.stack([
            torch.logsumexp(logits[candidates[label]], dim=0)
            for label in ("false", "true")
        ])
        if not torch.isfinite(masses).all():
            return None
        return float(torch.softmax(masses, dim=0)[1].item())
    return None
