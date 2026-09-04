"""MistSense — caption metrics for the error-explanation task (§4.1).

Paper §4.1 reports BLEU, ROUGE-L, and CIDEr on the explanation outputs.
Uses `sacrebleu` (BLEU), `rouge_score` (ROUGE-L), and `pycocoevalcap` (CIDEr).
"""
from __future__ import annotations

from typing import Sequence

from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer


def bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU-4 (sacrebleu default). Returns 0-100 scale."""
    assert len(predictions) == len(references)
    out = corpus_bleu(list(predictions), [list(references)])
    return float(out.score)


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean ROUGE-L F-measure over a list of (pred, ref) pairs."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    return float(sum(scores) / max(1, len(scores)))


def cider(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Corpus CIDEr-D. References must be lists-of-strings per pred; we wrap singles."""
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [r] for i, r in enumerate(references)}
    res = {i: [p if p.strip() else "."] for i, p in enumerate(predictions)}
    score, _ = Cider().compute_score(gts, res)
    return float(score)


def all_caption_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    return {
        "bleu": bleu(predictions, references),
        "rougeL": rouge_l(predictions, references),
        "cider": cider(predictions, references),
        "n": len(predictions),
    }
