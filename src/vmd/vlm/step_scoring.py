"""Read a distribution over recipe steps from the VLM's own generation.

The ``belief`` prompt lists executable steps as a lettered menu and asks for a
compact JSON whose FIRST field is the step letter. Because generation already
returns per-position logits (``output_scores=True``), the softmax over the
letter tokens at that position is a step likelihood for free -- no second
forward pass. Tokenisation is pinned: after the prefix ``{"step":"`` every
letter A--Y and the transit label ``-`` is exactly one token
(tests/test_step_label_table.py, run when the tokenizer is cached).
"""
from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vmd.tracking.belief import MENU_LETTERS, TRANSIT_LABEL

STEP_PREFIX = '{"step":"'      # compact JSON, no spaces: the tokenizer merges '": "' otherwise


class StepLabelTable:
    """Token ids of the menu letters (and ``-``) in the ``STEP_PREFIX`` context."""

    def __init__(self, tokenizer, max_steps: int = len(MENU_LETTERS)):
        self.letters: List[str] = list(MENU_LETTERS[:max_steps]) + [TRANSIT_LABEL]
        base = tokenizer(STEP_PREFIX, add_special_tokens=False)["input_ids"]
        self.prefix_ids: List[int] = list(base)
        self.token_ids: List[int] = []
        for L in self.letters:
            full = tokenizer(STEP_PREFIX + L, add_special_tokens=False)["input_ids"]
            new = full[len(base):]
            if list(full[: len(base)]) != list(base) or len(new) != 1:
                raise ValueError(f"letter {L!r} is not a single token after {STEP_PREFIX!r}: {full}")
            self.token_ids.append(int(new[0]))
        self.id_to_index = {t: i for i, t in enumerate(self.token_ids)}

    def ids_for(self, n_steps: int) -> List[int]:
        """Candidate ids for a recipe with ``n_steps`` executable steps + transit."""
        return self.token_ids[:n_steps] + [self.token_ids[-1]]


def locate_step_position(gen_ids: Sequence[int], table: StepLabelTable) -> Optional[int]:
    """Index in ``gen_ids`` of the token right after the ``{"step":"`` prefix.

    That token is normally a bare letter, but the tokenizer may merge the label
    with the closing quote (``-"``), so the caller must decode it rather than
    require table membership. Falls back to the first bare letter token.
    """
    ids = list(gen_ids)
    p = table.prefix_ids
    for k in range(len(p), len(ids)):
        if ids[k - len(p):k] == p:
            return k
    for k, t in enumerate(ids):
        if t in table.id_to_index:
            return k
    return None


def step_probs_from_scores(gen_ids: Sequence[int], scores, table: StepLabelTable,
                           n_steps: int, tokenizer=None) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """(probabilities over n_steps + transit, emitted menu index) or (None, None).

    ``scores`` is the tuple of per-position logits from ``generate``. If the
    emitted token is a merged form (label + quote), its logit is folded into the
    matching label with a log-sum-exp so the emitted label keeps its mass.
    """
    import torch
    k = locate_step_position(gen_ids, table)
    if k is None or k >= len(scores):
        return None, None
    logits = scores[k][0].float()
    cand_ids = table.ids_for(n_steps)                      # letters[:n] + transit
    cand = logits[torch.tensor(cand_ids, device=logits.device)].clone()
    tok_id = int(gen_ids[k])
    emitted = table.id_to_index.get(tok_id)
    if emitted is None:
        # merged token: decode and map its first character to a label
        piece = tokenizer.decode([tok_id]) if tokenizer is not None else ""
        first = piece.strip()[:1]
        if first in table.letters:
            j = table.letters.index(first)
            if j < n_steps or first == TRANSIT_LABEL:
                slot = n_steps if first == TRANSIT_LABEL else j
                cand[slot] = torch.logaddexp(cand[slot], logits[tok_id])
                emitted = slot
    else:
        if emitted == len(table.letters) - 1:
            emitted = n_steps                              # transit label -> extra slot
        elif emitted >= n_steps:
            emitted = None
    probs = torch.softmax(cand, dim=-1).cpu().numpy()
    return probs, emitted


def build_menu(descriptions: Sequence[str]) -> str:
    return "\n".join(f"{MENU_LETTERS[i]}. {d}" for i, d in enumerate(descriptions)) + \
        f"\n{TRANSIT_LABEL}. between steps / no recipe step in progress"


def compact_target(step_label: str, deviation: bool, mistake_class: str) -> str:
    """Training target and expected output format, step first."""
    return json.dumps({"step": step_label, "deviation": bool(deviation), "mistake_class": mistake_class},
                      separators=(",", ":"))
