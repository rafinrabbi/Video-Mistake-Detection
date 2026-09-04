# Venue recommendation

**Recommendation: ACL Rolling Review → EMNLP 2026 (main), with Findings as the
realistic floor.** Confirmed with you on 31 Aug 2026.

## Why ARR fits this paper

**The problem is published at *ACL venues*, not only CV ones.** The closest
methodological neighbour to this work — Storks et al., *Transparent and Coherent
Procedural Mistake Detection* — is **EMNLP 2025**. That paper asks a VLM to detect
procedural mistakes *and justify them*, and introduces NLI-based metrics for
rationale coherence. Our paper is the same shape of contribution: structured
prompting, a language-mediated plan representation, a structured-JSON output
contract, and a finding about reading confidence out of generation logits. An ARR
reviewer pool is the one most likely to recognise what is actually new here.

**The contribution is prompt/representation-side, not architecture-side.** We
train no new vision model. The method is: put the plan in the context, walk it as
a state machine, read the score from the logits, fine-tune with completion-only
masking. A CVPR/ICCV reviewer tends to read that as "prompt engineering plus
LoRA"; an ARR reviewer reads it as conditioning design, which is what it is.

**Rolling deadlines are worth a lot given the repo's state.** The one
load-bearing experiment (the graph ablation, ~24 h GPU) is not finished. ARR
cycles monthly, so the submission date is set by when that run lands rather than
by a fixed conference deadline — no need to either rush it or lose six months.

**The review cycle absorbs the paper's weaknesses.** Reviews plus a revision
window is exactly the right instrument for "macro-F1 is 0.185" and "one dataset,
one seed". A single-shot CV deadline is not.

**The taxonomy lives in the prompt.** Because there is no classifier head, the
class set can change without retraining. That is an argument that lands with an
NLP audience and is largely invisible to a CV one.

## What it costs you

- **Video-heavy work is a minority at *ACL venues.** Some reviewers will be less
  fluent in egocentric video benchmarks and evaluation protocols. Mitigate by
  writing §3 (problem formulation) and §5 (setup) so they stand alone, and by
  keeping the qualitative figure — it does a lot of work for a non-CV reader.
- **8-page limit.** The paper is currently 7 pages with two results sections
  still to be filled. Budget for it; the graph-ablation discussion and the
  qualitative figure both need room.
- **Mandatory Limitations section** (unnumbered, does not count against the page
  limit) — already drafted.
- **ARR requires a responsible-NLP / AI-use checklist** at submission. The
  disclosure and ethics sections are stubbed in red in the manuscript.

## Before you commit to a cycle

Submit only once **item 1 in `EXPERIMENTS_TODO.md` (the graph ablation) is
finished**. Without it, the paper's title claims the graph and its evidence
shows fine-tuning; that is the one criticism that would sink it, and it is
24 hours of GPU away rather than a redesign.

## Fallbacks, in order

1. **EMNLP Findings** — the natural landing spot if reviewers want a second
   dataset or stronger typing. No extra work needed.
2. **WACV 2027** — best pure-CV fit; explicitly welcomes rigorous applied
   systems. Expect harder questions on the vision side and a firm demand for a
   second benchmark.
3. **BMVC 2026** — note that Spurio et al. (a closely related VLM
   mistake-detection paper) was just accepted there, which is evidence of fit but
   also means your work will be read directly against it.
4. **CVPR/ICCV EgoVis or assistive-AI workshop** — highest acceptance
   probability and fastest, appropriate if you want the work archived quickly and
   extended later.

## What to do with the parts we cut

The adaptive-threshold work (`src/vmd/decision/threshold.py`, the τ-fitting
machinery, the falsification at full scale) is out of this manuscript by your
decision. It is not wasted: it is a self-contained negative result with a clean
mechanism analysis — a learned, interpretable, honestly-fit adaptive threshold
losing to a constant, with the fitted complexity weight coming out the wrong
sign. That is a workshop paper or a short-paper submission on its own if you ever
want it.

Note that the Contribution-B source (`src/vmd/decision/threshold.py`,
`src/vmd/vlm/uncertainty.py`, `scripts/cc4d_fit_and_eval_threshold.py`,
`tests/test_threshold.py`) was staged for deletion before this cleanup and is
removed from the working tree. It is recoverable from git history at commit
`141385c`, and the fitted parameters and per-window features survive in the saved
prediction records (`features`, `tau_adaptive`, `tau_fixed` fields).
