# Simulated pre-submission review — *Walk the Graph*

**Venue modelled:** ACL ARR → EMNLP 2026 · **Date:** 2026-09-01 · **Commit:** `2c41622`
**Read-only:** this document does not modify the manuscript.

**Reviewed as if R5/R6/R7 were present**, per instruction. Findings marked
**[PENDING-INDEPENDENT]** would survive even after the ablation lands — they are
not fixed by running the chain.

---

## Phase 0 — Panel configuration

Primary field: multimodal ML / video understanding. Secondary: NLP (structured
prompting, generative-model confidence). Paradigm: empirical systems paper with
a single causal claim. Maturity: near-submission, one blocking experiment open.

| Reviewer | Identity | Focus |
|---|---|---|
| EIC | ARR senior AE, multimodal | fit, claim calibration, decision |
| R1 Methodology | egocentric video eval, protocol design | selection discipline, statistics |
| R2 Domain | procedural mistake detection | positioning, baseline fairness |
| R3 Perspective | applied/assistive systems | deployment meaning of the metrics |
| DA Devil's Advocate | — | attribution, cherry-picking, dead claims |

---

## Verification pass (performed before review)

All 16 prose-bearing macros were recomputed from `outputs/decision_v2/*.json`:
**zero drift.** The generated-tables design works and removes a whole class of
error. Numerical integrity is not among the findings below.

---

## Phase 1 — Reviewer reports

### Devil's Advocate

**Strongest counter-argument.** *The paper's headline framing survives only by
choosing which comparisons to print.* Under the matched protocol the reproduced
baseline is not merely at parity on F1 — it is strictly better on the two
localisation quantities the abstract advertises as achievements. The abstract
says the system "localises the onset of 57 of 61 localisable error events at a
mean onset error of 60.4 s". The baseline localises **61 of 61** at **60.9 s**;
the zero-shot arm also reaches **61/61**. So on event coverage the proposed
system is last of the three, at an onset error indistinguishable from the
baseline's, and the abstract presents this as a selling point. Likewise "at
higher precision" is true (0.407 vs 0.325) and is printed, while recall — 0.589
vs 0.893, a 0.304 deficit — is not. Each individual sentence is defensible; the
selection is not. A reviewer who reads Table 1 before the abstract will feel
handled, and that impression is expensive.

**DA-1 [CRITICAL] — Broken transparency promise. [PENDING-INDEPENDENT]**
§4.4 states: *"we report calibration error alongside every result so that this
is visible rather than buried."* **No calibration number appears anywhere in the
paper.** ECE and Brier are absent from Table 1, from §5.4's metric list, and
from the prose. The artifacts contain them (ECE 0.490, Brier 0.500 for the main
row). The paper therefore makes a transparency claim it does not honour, about
the one metric where the system looks worst. Whatever the intent, this reads as
burying. Either report ECE/Brier, or delete the sentence — do not ship both.

**DA-2 [MAJOR] — A mechanism claim with no experiment, and none planned.
[PENDING-INDEPENDENT]**
The introduction asserts: *"the graph must be walked, not merely attached. A
static dump of all recipe steps tells the model nothing about position."* The
conclusion escalates it: *"The two design decisions that carry the result are
that the graph is walked as a state machine ... rather than dumped into the
context, and that the decision score is read from the model's logits."* No
experiment compares a walked pointer against a static dump of the step list.
Critically, **R5/R6/R7 do not test this either** — they vary graph presence and
graph content, not the walk. So the claim is unsupported today and will remain
unsupported after the chain finishes. Either run a static-dump arm (a fourth
prompt variant, one full test pass), or demote the claim from mechanism to
design description.

**DA-3 [MAJOR] — Attribution outruns the evidence on the second "design
decision" too.** The confidence-source result is +0.0058 binary F1
(deviation-only 0.4758 → 0.4816), which is far inside the ±0.08 bootstrap width
on this metric. §6.4 handles this honestly ("not that it rescues the binary
decision, but that it exists at all"), but the introduction bills it as "a
confidence-source study showing that the decision-token probability is what
makes threshold-based gating possible at all", and the conclusion calls it one
of two decisions that "carry the result". A quantity that moves F1 by 0.006
does not carry the result. The defensible claim is narrower and still
interesting: the verbalised field is constant, so it supports no ranking at all,
and the token probability is the only continuous signal available — evidenced by
AUC 0.636 vs an undefined ranking, not by the F1 delta.

**DA-4 [MAJOR] — "State-of-the-art" is not established.**
The abstract claims parity with "a reproduced state-of-the-art online detector".
One recent detector was reproduced. No comparison exists against ESTANet,
PREGO, TI-PREGO, EgoPER, or Seminara et al. on this benchmark. Parity with one
2025 system does not establish parity with the state of the art. Say "a
recent online detector" or "the closest published method to our setting".

**DA-5 [MODERATE] — The parity claim rests on a baseline the authors trained and
deliberately did not tune.** §5.5 states the baseline over-fires and that
tuning it away was declined because "polishing a baseline is not the
contribution". That is reasonable for a reference point, but the paper then
converts the reference point into a *parity claim* about the proposed method.
Its operating point (recall 0.893 at precision 0.325) is far from F1-optimal;
the decision layer was val-tuned, but the underlying detector's class weighting
was not. A reviewer will ask whether a properly weighted baseline exceeds 0.482.
The honest framing is that parity holds *against this reproduction*, with the
reproduction's limits stated in the same sentence.

**DA-6 [MODERATE] — The interval is wide enough to contain a loss.**
Δ = +0.005 with CI [−0.060, +0.059]. "Matches" is fair, but the data are equally
consistent with the baseline being 0.06 F1 better. State the width where the
claim is made, not only in the table.

**Non-defects.** The zero-shot row is characterised correctly (AUC 0.480, CI
straddles chance; high EDS attributed to indiscriminate firing). The
`missing_step` unreachability disclosure is exemplary. Selection discipline
(val-tuned, frozen, test touched once, per-arm layers) is genuinely above the
norm for this literature.

### R1 — Methodology

**Strengths.** Full-split evaluation with clip-level bootstrap, recordings as
the resampling unit, per-arm decision layers, and a re-gating layer that
reproduces GPU runs on CPU — this is unusually disciplined, and the
protocol-matched baseline re-tiling is the right way to handle a mismatched
comparison.

**M-1 [MAJOR] — Single seed.** Disclosed in Limitations, which is acceptable for
the headline gap (0.332, CI [0.246, 0.409]). It is *not* acceptable for the
per-class table: macro-F1 0.185 with classes at 0.060 and 0.082 cannot be
distinguished from run-to-run noise by any evidence in the paper. Either mark
the per-class table explicitly as single-run descriptive, or do not draw
conclusions from the ordering of the small classes.

**M-2 [MODERATE] — Undisclosed exclusion.** `ablation.md` records "untyped
dropped: 1" — one `Other`-only step removed from typed metrics. The paper says
"no subsetting" and "zero recordings dropped". Both are true at recording level;
the typed-metric exclusion should still be stated in §5.2 beside the 69
non-localisable segments.

**M-3 [MODERATE] — Two decision rules, one table.** Binary F1 gates on smoothed
score > 0 while localisation requires *m* consecutive fired windows. This is
disclosed and deliberate, but it means EDS and F1 in the same row describe
different decision procedures. Add a one-line reminder in the Table 1 caption.

**M-4 [MINOR] — McNemar over 23,696 correlated windows** is reported as
secondary with the correlation caveat. Correct, but p = 2.8×10⁻⁸ from
non-independent units invites misreading; consider dropping it since the paired
bootstrap already carries the claim.

### R2 — Domain

**D-1 [MAJOR] — The typed claim is the paper's weakest load-bearing element.**
Macro-F1 0.185 (0.212 excluding `missing_step`), with `temperature_error` at 89
windows and `missing_step` at 2. The paper is admirably clear that this is
partial. But "additionally naming the mistake type" appears in the abstract and
in the contributions as a differentiator from the baseline. With two classes
unmeasurable and four below 0.17, that differentiator is thin. Recommend:
present typing as *evidence the formulation is richer*, explicitly not as a
capability claim.

**D-2 [MODERATE] — Graph conditioning is thinner than the framing.** §4.2's
disclosure (only steps and precedence edges; objects/states/variations empty on
all 24 recipes) is honest and well placed. But the paper's own framing — "the
plan and the percept meet in the same medium" — describes something richer than
step description + valid next steps + position. Keep the disclosure adjacent to
the framing, not two subsections later.

**D-3 [MODERATE] — Seminara et al. handled well; extend the same treatment.**
The explanation of why no head-to-head exists (they use CC4D for graph
generation; mistake detection is on Assembly101-O/EPIC-Tent-O from action
sequences) is exactly right. ESTANet and EgoPER get no equivalent treatment —
one sentence each on why they are not comparison rows.

### R3 — Perspective

**P-1 [MAJOR] — At the reported operating point the system is not deployable,
and the paper does not say so.** Precision 0.407 means roughly three false
alarms for every two true ones. For the assistive-AR and training use cases
named in the introduction, an assistant interrupting a cook that often is worse
than silence. The ethics statement gestures at this ("would produce a
substantial rate of false accusations if deployed naively"), but the
introduction's motivation and the results never meet. Add one paragraph
connecting them: what precision would be needed, and how far away is this.

**P-2 [MODERATE] — Onset error of 60 s is reported without interpretation.**
Is 60 s useful? For a recipe step lasting 30 s it is worthless; for a 10-minute
bake it may be fine. Give the reader the step-duration distribution so the
number means something.

**P-3 [MINOR]** — The cucumber example in the introduction is excellent and
does real work. Keep it.

### EIC

Fit for ARR is good: structured prompting, a generative-confidence finding, and
a language-mediated plan representation, with EMNLP 2025 precedent
(Storks et al.). Writing is clear and unusually candid — the `missing_step` and
graph-thinness disclosures build real credibility.

That candour makes DA-1 and DA-2 more damaging than they would be in a
carelessly written paper: the manuscript trains the reader to trust its
disclosures, then breaks one (calibration) and over-attributes on another (the
walk). Both are cheap to fix and neither touches a number.

The paper cannot be assessed for acceptance until R5/R6/R7 land — the single
claim is currently unevidenced.

---

## Phase 2 — Editorial decision

**Decision: Major Revision** (provisional; blocked on the pending ablation).

Per Checkpoint Rule 4, the Devil's Advocate CRITICAL finding (DA-1) precludes
Accept regardless of the ablation outcome.

**Consensus across ≥3 reviewers:** claim calibration outruns evidence in the
abstract/introduction/conclusion, while the body is honest — the defects are
concentrated in the framing, not the science.

**Disagreement:** R3 wants the deployability gap foregrounded; EIC considers it
a Limitations matter. Editor sides with R3 *only* to the extent of one
paragraph, since the introduction itself raises the applications.

### Revision roadmap (priority order)

| # | Finding | Action | Cost |
|---|---|---|---|
| 1 | DA-1 | Report ECE/Brier in Table 1, or delete the §4.4 sentence promising them | 15 min |
| 2 | DA-2 | Demote "walked, not merely attached" to design description, or run a static-dump arm | 15 min / 1 pass |
| 3 | DA-3 | Rewrite the confidence claim around AUC and the absence of any ranking, not the +0.006 | 20 min |
| 4 | DA-4 | "recent" not "state-of-the-art" | 2 min |
| 5 | DA-6, DA-5 | State the CI width and the untuned-baseline caveat where parity is claimed | 15 min |
| 6 | DA-cherry | Abstract must state recall and event coverage alongside precision | 20 min |
| 7 | P-1 | One paragraph connecting the motivation to precision 0.407 | 30 min |
| 8 | D-1 | Frame typing as formulation evidence, not capability | 15 min |
| 9 | M-1, M-2 | Mark per-class table single-run; disclose the 1 untyped-dropped step | 15 min |
| 10 | — | R5/R6/R7, then the abstract/intro ablation sentences | blocked on GPU |

Items 1–9 are ~2.5 hours of writing, need no GPU, and change no number.
