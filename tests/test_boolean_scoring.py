import math

import pytest
import torch

from vmd.vlm.boolean_scoring import deviation_probability
from vmd.vlm.qwen25vl import _parse_output


class Tokenizer:
    pieces = ['{"deviation":', 'true', 'false', ' true', ' false', ',', 'unrelated', '"rationale":']

    def encode(self, text, **kwargs):
        return [self.pieces.index(text)]

    def decode(self, ids):
        return ''.join(self.pieces[i] for i in ids)


@pytest.mark.parametrize("emitted", [1, 2, 3, 4])
def test_score_uses_same_positive_polarity_with_other_vocabulary_mass(emitted):
    # Boolean masses: true .02+.03, false .10+.05; rest is NOT positive evidence.
    logits = torch.tensor([[math.log(p) for p in [.1, .02, .10, .03, .05, .1, .5, .1]]])
    score = deviation_probability(Tokenizer(), [0, emitted, 5], [logits] * 3)
    assert score == pytest.approx(.25)


def test_unlocatable_boolean_has_no_confidence_fallback():
    logits = torch.zeros((1, 8))
    assert deviation_probability(Tokenizer(), [7, 1], [logits] * 2) is None


@pytest.mark.parametrize("raw", ['oops', '[]', '{"deviation":"false","mistake_class":"no_mistake"}',
                                 '{"deviation":false,"mistake_class":"unknown"}'])
def test_parse_failure_is_explicit_and_not_a_negative_score(raw):
    pred = _parse_output(raw, .8)
    assert pred.confidence is None and not pred.parse_ok
    assert pred.score_status == "parse_error"


def test_missing_score_does_not_use_verbalized_confidence():
    raw = '{"deviation":false,"mistake_class":"no_mistake","confidence":0.9}'
    pred = _parse_output(raw)
    assert pred.parse_ok and pred.confidence is None
    assert pred.score_status == "missing_boolean_score"
    assert _parse_output(raw, .2).confidence == .2


def test_real_cached_tokenizer_handles_compact_and_spaced_json():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    for raw in ['{"deviation":true}', '{"deviation": false}', '{"step":"A","deviation":false}']:
        ids = tok.encode(raw, add_special_tokens=False)
        logits = torch.full((1, len(tok)), -10.0)
        for text in ('true', ' true'):
            logits[0, tok.encode(text, add_special_tokens=False)[0]] = 1.0
        for text in ('false', ' false'):
            logits[0, tok.encode(text, add_special_tokens=False)[0]] = 2.0
        assert deviation_probability(tok, ids, [logits] * len(ids)) == pytest.approx(1 / (1 + math.e))
