# Procedural Mistake Detection — Detailed Progress Report

*Target venue: EMNLP 2026 via ACL ARR. Benchmark: CaptainCook4D (NeurIPS 2024 D&B).*
*Generated 2026-06-18. This report is deliberately conservative: it separates*
*what is **done and verified** from what is **in progress** or **unproven**.*

---

## 0. Honest one-paragraph status

We detect procedural mistakes in egocentric cooking video. Two intended
contributions: **(A)** a procedural-graph-conditioned VLM (Qwen2.5-VL-7B, LoRA
fine-tuned) that reasons clip-by-clip, and **(B)** an uncertainty-aware adaptive
decision threshold τ. This cycle we (1) migrated off the wrong dataset (EgoOops)
onto the intended benchmark (CaptainCook4D) with its **native 8-class** taxonomy,
(2) re-trained Contribution A, (3) found and fixed several evaluation-metric
bugs, (4) found Contribution B was inert (the model emitted a constant
confidence) and **made the adaptive-threshold fit operable again** by reading
confidence from output-token probabilities, and (5) launched the full
evaluation. **Two honesty caveats up front:** (i) the headline full-test (109-clip,
8-class) numbers **do not exist yet** — only the validation passes have run; and
(ii) making the threshold fit *succeed* is **not** the same as showing adaptivity
*helps* — on the completed validation eval the adaptive threshold is currently
**no better than (slightly worse than) a fixed threshold**. Whether Contribution
B provides a real benefit is an **open question** pending the test passes.

---

## 1. Problem and intended contributions

**Task.** For each moment of a multi-step cooking video, decide whether the
person is making a mistake and, if so, its *type*.

**Contribution A — Procedural Graph-Guided VLM Reasoning.** Represent the recipe
as a procedural graph (steps + precedence edges). At each window, feed
Qwen2.5-VL-7B the frames + graph context (current expected step, valid next
steps, required objects/states) and have it emit structured JSON: `deviation`,
`mistake_class`, `confidence`, rationale, uncertainty reason. LoRA fine-tuned on
the benchmark.

**Contribution B — Uncertainty-Aware Adaptive Decision Threshold.** Make the
flagging threshold τ a per-window linear function of interpretable signals,
fit by logistic regression on validation:
> τ = clip( α·complexity + β·(1/variants) + γ·u + δ·(1−graph_conf) + ε·hist + bias , 0, 1 )
A window is flagged when the model reports a deviation **and** its confidence
clears τ. **Design intuition:** τ should *rise* when the model is unreliable
(complex task, high uncertainty, weak graph match). *See §8 — the data-driven fit
currently contradicts part of this intuition, which is an open issue.*

---

## 2. The dataset: CaptainCook4D

Egocentric (GoPro) cooking videos with step + error annotations and ground-truth
task graphs (arXiv 2312.14556, NeurIPS 2024 D&B). 360p release (~31 GB, ~94.5 h).
Verified counts (`data/raw/captaincook4d/INSPECTION.md`):

| Quantity | Value |
|---|---|
| Recordings | **384** (220 with ≥1 error) |
| Recipes / task graphs | **24** |
| Step-segments (total) | **5700** |
| Error step-segments | **1964** |
| Error *instances* (a step may carry several) | **2574** |
| Temporally localizable (start ≥ 0) | 1683 |
| **Non-localizable (start = −1, mostly Missing-Step)** | **281** |

**Official split:** train **213** / val **62** / test **109**.

**Native 8-category taxonomy** (instances): Order 795, Technique 502, Preparation
410, Measurement 331, Missing Step 285, Timing 177, Temperature 66, Other 8.

**Terminology** (a reviewer will ask): a *step-segment* (5700) is a labeled time
window; an *error-segment* (1964) is a step-segment with ≥1 mistake; an
*error-instance* (2574) is one mistake tag — multi-error steps make 2574 > 1964.

---

## 3. Why we switched datasets (EgoOops → CaptainCook4D)

EgoOops (~30–50 clips) is too small: the MistSense baseline collapsed on it
(test F1 ≈ 0). CaptainCook4D has ~5700 supervised step-segments; the same
baseline trains to a working detector there. This cycle we committed to the
switch and **removed EgoOops entirely** (an explicit decision — it was the wrong
benchmark choice).

---

## 4. The 8-class taxonomy decision (and its built-in limitation)

While on EgoOops we maintained a lossy 8→6 remap. Dropping EgoOops removed the
reason for it, so we adopted CaptainCook4D's **native taxonomy**, dropping only
the `Other` catch-all (8 garbage instances) → **8 classes**:
```
no_mistake, order_error, technique_error, preparation_error,
measurement_error, missing_step, timing_error, temperature_error
```
Single source of truth: `src/vmd/data/mistake_classes.py`.

**Critical limitation we must report:** our detector is a **windowed**
classifier, but `missing_step` errors are exactly the **non-localizable**
(start = −1) segments — a step that *never happened* produces no time window.
So the windowed pipeline **structurally cannot emit `missing_step`**; in every
eval file its support = 0 and per-class F1 = 0.0. **In practice the windowed
detector covers 7 of the 8 classes.** We therefore must report macro-F1 **both
with and without** `missing_step`, and treat missing-step detection (absence
reasoning over the whole sequence) as out of scope / future work.

Per-class **validation** support our loader produced (online): no_mistake 9956,
order_error 2270, measurement_error 629, preparation_error 557, timing_error 424,
technique_error 287, **temperature_error 6, missing_step 0**. Three classes have
support ≤ ~20 windows → their per-class F1 is statistically unstable (see §11).

---

## 5. System architecture

```
CC4D video → sampler (causal online / full offline)
  → procedural graph (gold CC4D graph or LLM-built)
  → qwen25vl (graph-conditioned prompt → JSON: deviation, class, confidence)
  → uncertainty (composite score)
  → AdaptiveThreshold τ  (Contribution B)
  → postprocess (median smoothing + start-of-deviation localization)
  → metrics (binary/macro/per-class F1, AUC, EDS, ECE, Brier)
```
Two protocols, intended to be reported side by side: **online** (causal sliding
window) and **offline** (full video). *Status: only online artifacts exist so
far; offline is produced by the still-running test passes.*

---

## 6. Chronological work log (this cycle)

### Step 0 — Snapshot baseline `ba47567`
git-init, committed pre-refactor state (1713 files), tag `pre-7class-refactor`,
backed up the old 6-class adapter + predictions.

### Step 1 — Scope investigation (4 parallel agents)
Mapped all 119 EgoOops references, the hardcoded-6-class sites, the loader/remap
call sites, and A/B state → produced an exact edit list.

### Step 2 — Drop EgoOops, 8-class native taxonomy `208bb5c` (26 files, +196 / −847)
New `mistake_classes.py`; deleted the STRICT/LENIENT remap module; rewrote the
CC4D loader + Qwen prompt + class validator for 8 classes; added `configs/cc4d.yaml`;
deleted `configs/egoops.yaml`, 6 EgoOops-only scripts under `scripts/` (+1
baseline sanity script), and the 11 GB `data/raw/egoops/`. Kept the
`EgoOopsClip`/`MistakeAnnotation` dataclass names (rename would churn 10+ files
for no gain — explicit decision). Tests green.

### Step 3 — Re-train Contribution A (LoRA), 8-class
Qwen2.5-VL-7B, LoRA r=32, α=64, dropout 0.05 on q/k/v/o/gate/up/down. **95M
trainable params (1.13%).** **Completion-only LM cross-entropy** (generate the
target JSON; prompt+image tokens masked) — so the class set is *not* a fixed
classifier head. 10,000 class-balanced windows (4951+/5049−), stride 4 s, 2
epochs, ~3 h on one RTX PRO 6000. Final train loss ≈ 0.017. Adapter:
`outputs/lora_cc4d_8class/` (**364 MB**).

### Step 4 — Phase 9: collect val preds + fit τ → **caught the B failure** (§8).

### Step 5 — Harden the eval harness `f32fd4b` (9 files, +518 / −36)
Driven by an adversarial multi-agent review of `src/vmd/eval/`. Fixed 7 real
issues (§7). 48 unit tests pass (was 44).

### Step 6 — Make Contribution B operable `141385c` (2 files, +121 / −9)
Token-probability confidence (§8).

### Step 7 — Launch full eval (running; only validation passes done so far) (§9).

---

## 7. Evaluation-metric bugs found and fixed (verified)

1. **Added per-class F1** (`average=None`, explicit labels so a never-predicted
   class scores 0.0, not dropped) + support counts.
2. **Fixed ECE/Brier**: were calibrating against the *positive label* instead of
   *prediction correctness*; now use `correctness = (pred == true)`.
3. **Made fixed vs adaptive an honest comparison**: each τ variant now re-gates
   its own class stream and uses its own score stream (previously the non-binary
   metrics were byte-identical between them).
4. **Added localization** (EDS, start-time MAE) against the earliest localizable
   gold event (online only).
5. **Added explicit coverage accounting** (clips/windows/graph-missing/dropped/
   localizable) so nothing is silently capped; surfaced the 69 non-localizable +
   1 dropped test segments.
6. **Threshold fit never aborts the pipeline**: always writes params + a
   `.meta.json` sidecar with `fitted: true/false` + reason, so a collapsed fit is
   auditable.
7. **Orchestrator + aggregator** scripts with a hard assert on per-class key
   alignment and honest footnotes.

These are genuine correctness fixes, independent of the model results.

---

## 8. Contribution B: failure, partial fix, and what is still unproven

**Failure (verified).** On the full validation set (14,129 windows) the LoRA
model emitted `confidence = 0.9` on **every** window — a constant. The logistic
threshold fit produced a **negative** confidence coefficient (`w_conf = −0.13`)
and correctly fell back to default parameters. With constant confidence, adaptive
and fixed τ make identical decisions — B was inert. Root cause: the LoRA training
targets hard-coded the `confidence` field, so the model memorized it.

**Engineering fix (verified to make the fit succeed).** We read confidence from
the model's **output token probabilities** — the softmax probability of its
`deviation` decision token — instead of the verbalized text
(`vlm/qwen25vl.py`, commit `141385c`). On the full validation set the fit now
**succeeds**: `fitted: true`, `w_conf = +1.447`, real (non-default) weights.

**Three things this does NOT yet show — stated plainly:**

1. **Adaptive does not beat fixed (current evidence).** On the completed
   validation eval (`outputs/full_eval_8class/lora_val_eval.json`, online), the
   adaptive threshold is **worse** than a fixed 0.5 on every headline metric:

   | metric | adaptive | fixed |
   |---|---|---|
   | binary F1 | 0.460 | **0.479** |
   | AUC | 0.619 | **0.630** |
   | EDS (earliness) | 0.560 | **0.747** |
   | start-MAE (s) ↓ | 81.4 | **61.8** |
   | events detected | 30/36 | **34/36** |

   (These validation numbers use the *default* adaptive parameters during
   collection; the *fitted*-τ-vs-fixed comparison is what the test passes will
   produce. Either way, there is **no evidence yet that adaptivity helps**, and
   the current signal is neutral-to-negative.)

2. **The fitted weights contradict the design intuition.** The fitted complexity
   weight is **α = −0.405 (negative)** — i.e. the data says *more complex tasks
   get a LOWER bar*, the opposite of the stated "τ rises with complexity." Fitted
   params: α −0.405, β 0.447, γ 0.868, δ 0.089, ε 0.525, bias 0.481
   (`threshold_params_lora_8class.json`). This must be explained or the feature
   re-examined — we do **not** hide it.

3. **The new confidence is informative but poorly calibrated.** It varies with
   correct polarity (low on no-mistake, high on confident deviations), which
   unblocks the fit — but its calibration is bad (**ECE ≈ 0.47, Brier ≈ 0.48** on
   val). So "verbalized confidence is degenerate; token-probability is a usable
   signal" is defensible; "it is well-calibrated" is **not** — calibration is
   future work.

**Honest takeaway:** the *mechanism* of Contribution B is now operable and the
underlying insight (verbalized VLM confidence is degenerate; token-decision
probability carries signal) is real and reportable. Whether the adaptive
threshold *improves detection* over a fixed one is **unresolved** and currently
unsupported.

---

## 9. Full evaluation — actual state

`scripts/cc4d_full_eval.sh`, all 109 test clips, no subsetting, stride 4 s. Four
passes:

| Pass | Work | Status |
|---|---|---|
| A1 | LoRA full-val collect → fit τ_LoRA | **done** (fit succeeded; see §8) |
| A2 | zero-shot full-val collect → fit τ_zeroshot | **running** |
| B  | zero-shot all-109 test, online+offline | **not started** |
| C  | LoRA all-109 test, online+offline | **not started** |

**There is currently no full-test (109-clip, 8-class) result for either model,
and no fitted-τ-on-test comparison.** The ablation table
(`outputs/full_eval_8class/ablation.{json,md}`) does not exist yet. It will have
rows {zero-shot, LoRA} × {fixed, fitted τ} + MistSense, and columns binary/macro/
per-class F1, AUC, precision, recall, ECE, Brier, and (online) EDS + start-MAE.

---

## 10. Results that exist today (with comparability caveats)

| System | Setting | Number | Caveat |
|---|---|---|---|
| MistSense (baseline) | CC4D test, online | f1_mistake **0.495** @ τ0.70 | **Not directly comparable**: 2 fps, different window definition, **binary-only** (no typing), and τ chosen by post-hoc sweep (oracle). |
| Qwen zero-shot, graph-cond | CC4D test, **16-clip subset** | binary_f1 0.176 | Tiny subset; overconfident (ECE 0.646). |
| Qwen + LoRA, **6-class (superseded)** | test subset | binary_f1 0.535 | Old taxonomy; superseded. |
| Qwen + LoRA, 8-class | **16-clip subset**, default τ | binary_f1 0.529 | Subset only; **default unfitted τ** (`fitted: false`). |

**We must NOT present "0.176 → 0.535" as evidence for the current contribution:**
0.535 is the superseded 6-class adapter on a 16-clip subset, and the only 8-class
number (0.529) is also a subset with an unfitted threshold. The defensible
zero-shot→LoRA comparison requires the full-test passes (B, C) on identical
clips/protocol — pending.

---

## 11. Threats to validity / open questions (what a committee will ask)

1. **No statistical validity yet.** All deltas are single-run point estimates, no
   confidence intervals, no significance test, one seed. The adaptive-vs-fixed
   gap (~0.02 F1) is within plausible noise *and currently in the wrong
   direction*. **Plan:** per-clip bootstrap CIs and a paired test (clip-level
   bootstrap / McNemar on window decisions) for (a) zero-shot vs LoRA and (b)
   fixed vs adaptive τ.
2. **Severe class imbalance.** macro-F1 (~0.19 on val) is dominated by
   near-zero rare-class scores (temperature_error support 6, timing_error,
   missing_step 0). Report support with every per-class F1; also report macro-F1
   over the well-supported subset and a frequency-weighted F1.
3. **`missing_step` unreachable** by the windowed detector (§4) — report macro
   with and without it; pre-empt "how do you detect a step that never occurred?"
4. **Negative fitted α** contradicts the complexity intuition (§8) — explain or
   re-examine the feature.
5. **Poor calibration** of the token-probability confidence (ECE ≈ 0.47) — §8.
6. **Baseline comparability** — MistSense is binary-only, 2 fps, oracle-τ;
   either re-run under our 1 fps / stride-4 windowing or footnote loudly (§10).
7. **No related-work positioning yet** for either contribution (graph-guided
   video reasoning; adaptive/conformal/calibrated thresholding). Need to justify
   logistic-on-interpretable-features over Platt scaling / conformal / per-class
   thresholds, and token-prob confidence over a trained calibration head.

---

## 12. What is pending / next steps

1. **Finish the full eval** (~33 h remaining) → the first real full-test ablation.
2. **Statistical layer**: bootstrap CIs + paired significance tests (§11.1).
3. **Resolve B**: explain/fix the negative-α issue; decide whether adaptivity
   helps once fitted-τ-on-test exists; if not, reframe B (e.g. calibration of
   token-prob confidence, or per-class thresholds) honestly.
4. **Docs (Phase 11)**: `CLAUDE.md`, `RESULTS_SUMMARY.md`, `pipeline_diagram.md`
   still mention EgoOops/6-class — update to 8-class CC4D-only.
5. **Ablations**: graph-conditioned vs unconditioned; gold vs LLM-built graph.

---

## 13. Reproducibility

**Commits:** `ba47567` snapshot · `208bb5c` drop EgoOops + 8-class (26f, +196/−847)
· `f32fd4b` harden eval harness (9f, +518/−36) · `141385c` token-prob confidence
(2f, +121/−9).

**Key files:** `src/vmd/data/mistake_classes.py` (taxonomy) ·
`src/vmd/data/captaincook4d.py` (loader) · `src/vmd/vlm/qwen25vl.py` (VLM +
token-prob confidence) · `src/vmd/decision/threshold.py` (AdaptiveThreshold +
fit) · `src/vmd/eval/{online,offline,metrics}.py` · `scripts/cc4d_lora_train.py`
· `scripts/cc4d_full_eval.sh` + `cc4d_build_ablation.py` ·
`data/raw/captaincook4d/INSPECTION.md`.

```bash
pytest tests/ -v                                    # 48 tests, no GPU/data
python scripts/cc4d_lora_train.py --output_dir outputs/lora_cc4d_8class
bash scripts/cc4d_full_eval.sh                      # full ablation (~38h)
```

---

## 14. Anticipated questions (and honest answers)

- **Does the adaptive threshold help?** *Unknown yet.* The fit now succeeds, but
  on validation it does not beat a fixed threshold. The test-set comparison is
  pending; we will not claim a win we cannot show.
- **Your fitted model lowers the bar for complex tasks — why?** Open issue; the
  data-driven α is negative, contradicting our design intuition. We report it and
  will investigate (possible feature mis-sign or a real effect).
- **How do you detect a missing step?** We currently don't — it's non-localizable
  and out of scope for the windowed detector; we report macro-F1 with/without it.
- **Is the MistSense baseline measured the same way?** No — it's binary-only,
  2 fps, oracle-τ. We footnote this and plan a matched re-run.
- **Are your differences significant (n=109)?** Not yet tested; bootstrap CIs +
  paired tests are the immediate next step.
- **Did fine-tuning fix the class space in weights?** No — completion-only text
  generation; the class set lives in the prompt/targets, not a classifier head.
- **Why is the verbalized confidence useless?** The LoRA targets hard-coded it
  (constant 0.9); the model memorized it. Token-decision probability is the
  signal we use instead.
```
