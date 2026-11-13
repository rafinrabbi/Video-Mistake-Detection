"""Letter tokens are single tokens in the compact-JSON context; position lookup works.

Needs the Qwen2.5-VL tokenizer in the local HF cache; skipped otherwise.
"""
import os

import numpy as np
import pytest

from vmd.tracking.belief import MENU_LETTERS
from vmd.vlm.step_scoring import (STEP_PREFIX, StepLabelTable, build_menu, compact_target,
                                  locate_step_position, step_probs_from_scores)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
except Exception:                       # pragma: no cover - environment dependent
    tokenizer = None

needs_tok = pytest.mark.skipif(tokenizer is None, reason="Qwen tokenizer not cached")


@needs_tok
def test_every_letter_is_one_token_after_prefix():
    table = StepLabelTable(tokenizer)
    assert len(table.token_ids) == len(MENU_LETTERS) + 1
    assert len(set(table.token_ids)) == len(table.token_ids)


@needs_tok
def test_locate_and_probs_from_generated_sequence():
    import torch
    table = StepLabelTable(tokenizer)
    target = compact_target("C", True, "order_error")
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    k = locate_step_position(ids, table)
    assert k is not None and tokenizer.decode(ids[:k]) == STEP_PREFIX
    vocab = max(table.token_ids) + 5
    scores = tuple(torch.full((1, vocab), -10.0) for _ in ids)
    scores[k][0, table.token_ids[2]] = 2.0      # C
    scores[k][0, table.token_ids[0]] = 1.0      # A
    probs, emitted = step_probs_from_scores(ids, scores, table, n_steps=5)
    assert emitted == 2 and probs.shape == (6,) and np.argmax(probs) == 2
    assert probs.sum() == pytest.approx(1.0)


@needs_tok
def test_merged_transit_token_is_recovered():
    import torch
    table = StepLabelTable(tokenizer)
    # the model tends to emit '-"' as one token; the sequence below reproduces it
    ids = tokenizer('{"step":"-","deviation":false}', add_special_tokens=False)["input_ids"]
    k = locate_step_position(ids, table)
    assert k is not None and tokenizer.decode(ids[:k]) == STEP_PREFIX
    vocab = max(max(table.token_ids), max(ids)) + 5
    scores = tuple(torch.full((1, vocab), -10.0) for _ in ids)
    scores[k][0, ids[k]] = 3.0
    probs, emitted = step_probs_from_scores(ids, scores, table, n_steps=5, tokenizer=tokenizer)
    assert emitted == 5 and np.argmax(probs) == 5


def test_menu_and_target_format():
    menu = build_menu(["Take mug", "Add flour"])
    assert menu.splitlines()[0] == "A. Take mug" and menu.splitlines()[-1].startswith("-. ")
    assert compact_target("-", False, "no_mistake") == '{"step":"-","deviation":false,"mistake_class":"no_mistake"}'
