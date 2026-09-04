# Pipeline Summary — Procedural Mistake Detection in Egocentric Video

*State of the repo as of the completed Phase-10 full ablation
(`outputs/full_eval_8class/`, finished 2026-06-20).*
*Companion docs: `PROGRESS_REPORT.md` (narrative + threats to validity),
`PROJECT_REPORT.md`, `data/raw/captaincook4d/INSPECTION.md` (dataset audit).*

> **Status headline.** The full 109-clip test ablation is **done** — it is no
> longer "running" as `PROGRESS_REPORT.md` states. Contribution A works and is
> large (zero-shot 0.036 → LoRA **0.471** binary F1, online). Contribution B
> **does not work**: the adaptive threshold is worse than a fixed one on every
> metric, on every row, in both protocols. That is the central open problem.

---

## 1. What the system does

Given an egocentric (head-camera) video of someone cooking a recipe, decide —
window by window, causally — whether the person is deviating from the recipe,
and if so which of 7 detectable mistake types it is. The recipe is supplied as a
**procedural graph** (steps + precedence DAG), and the detector is a
graph-conditioned VLM rather than a trained classifier head.

**Two intended contributions:**

- **A — Procedural Graph-Guided VLM Reasoning.** Qwen2.5-VL-7B, LoRA
  fine-tuned, sees frames *plus* the graph context (expected current step,
  valid next steps, required objects/states, allowed variations) and emits
  structured JSON.
- **B — Uncertainty-Aware Adaptive Decision Threshold.** The confidence bar τ a
  window must clear varies per window as a linear function of interpretable
  signals, fit on validation.

---

## 2. End-to-end pipeline

```
CaptainCook4D recording (.mp4, 360p)
  │
  ├─ data/captaincook4d.py ────── loader: recordings → EgoOopsClip + MistakeAnnotation
  ├─ data/captaincook4d_graphs.py ─ gold CC4D task graph → ProceduralGraph (cached)
  │
  ▼
 data/sampler.py            4 s windows, stride 4 s, 1 fps  → Clip(frames, true_label, true_class)
  ▼
 vlm/qwen25vl.py            graph-conditioned prompt + frames → JSON
  │                         {deviation, mistake_class, confidence, rationale,
  │                          uncertainty_reason, observed_action/objects}
  │                         + token-probability confidence  ← overrides verbalized confidence
  ▼
 vlm/uncertainty.py         composite u = 0.6·(1−conf) + 0.4·reason_penalty ; history flip rate
  ▼
 decision/threshold.py      τ = clip(α·complexity + β·(1/variants) + γ·u
                                    + δ·(1−graph_conf) + ε·hist + bias, 0, 1)
                            fire ⇔ deviation ∧ confidence ≥ τ      [Contribution B]
  ▼
 decision/postprocess.py    median smoothing (k=3) + start-of-deviation (min_consecutive=2)
  ▼
 eval/metrics.py            binary F1, macro-F1, per-class F1 + support, AUC,
                            precision/recall, ECE, Brier, EDS, start-MAE
```

Orchestrated by `eval/online.py` (causal) and `eval/offline.py` (full-video),
driven by `scripts/cc4d_zeroshot_eval.py` (which also runs the LoRA model via
`--lora_adapter`), and chained end-to-end by `scripts/cc4d_full_eval.sh`.

### Stage detail

| Stage | File | What actually happens |
|---|---|---|
| Load | `data/captaincook4d.py` | 384 recordings → clips; `recording_id "1_10"` → activity 1 → `recipe_key` → graph stem. Non-localizable errors (`start=-1`) kept as `start=end=-1` so windowed labeling skips them. `Other` tag → untyped-dropped. |
| Graphs | `data/captaincook4d_graphs.py` | Gold CC4D DAG (`task_graphs/<recipe>.json`) → `ProceduralGraph` with START/END kept as `is_optional` sentinels. **29 cached graphs**, 0 missing at eval time. |
| Sample | `data/sampler.py` | Causal tiling `t += stride` while `t + clip_len ≤ duration`; window label = mistake covering the window **midpoint**. |
| Reason | `vlm/qwen25vl.py` | Prompt template `vlm/prompts/reason.txt`; greedy decode (`do_sample=False`), `output_scores=True`. Output JSON is regex-extracted and class-validated against `MISTAKE_CLASSES`; unparseable → safe `no_mistake` fallback. |
| Confidence | `vlm/qwen25vl.py::_deviation_token_confidence` | Softmax probability of the `true`/`false` token emitted after `"deviation":`, with a geometric-mean-token-probability fallback. **This replaces the verbalized `confidence`,** which the LoRA model emits as a constant 0.9. |
| Gate | `decision/threshold.py` | Per-window τ recomputed from live uncertainty + last-3 history. Fit by logistic regression on deviation-flagged val windows; `α..ε = −w_i/w_conf`, `bias = −b/w_conf`; requires `w_conf > 0` or it falls back to defaults and records why in a `.meta.json` sidecar. |
| Post | `decision/postprocess.py` | Median-smooth the gated score stream; localize first run of ≥2 consecutive fired windows. |
| Score | `eval/metrics.py` | Explicit `labels=` so a never-predicted class scores 0.0 (not dropped); ECE/Brier computed against `correctness = (pred == true)`. |

**Step tracking differs by protocol** — this is the real online/offline
distinction, since `offline_clips` just delegates to `online_clips`:
- *Online:* `current_step_idx` advances by 1 whenever the model reports **no**
  deviation (graph-following state machine).
- *Offline:* step index = window index `i`, clamped — so for any clip longer
  than `len(graph.steps)` windows it **saturates at the final step**, which is
  most clips. Offline conditioning is therefore weaker, and its numbers are
  correspondingly lower. Documented in footnote 4 of `ablation.md`.

---

## 3. Data

CaptainCook4D (NeurIPS 2024 D&B), 360p, ~94.5 h — the sole benchmark; EgoOops
was removed entirely in `208bb5c`.

| Quantity | Value |
|---|---|
| Recordings / videos on disk | 384 / **384** |
| Recipes (task graphs) | 24 (29 cached graph files) |
| Step-segments | 5700 (1964 with ≥1 error; 2574 error instances) |
| Non-localizable (`start=-1`) | 281 |
| Split (recordings) | train 213 / val 62 / test **109** |
| Eval windows | val **14,129** · test **23,696** (0 clips dropped, 0 graphs missing) |

**Taxonomy** (`data/mistake_classes.py`, single source of truth) — CC4D's native
8 minus the `Other` catch-all:

```
no_mistake  order_error  technique_error  preparation_error
measurement_error  missing_step  timing_error  temperature_error
```

Multi-error steps are reduced by `PRIORITY` (structural errors outrank
execution errors).

**Structural limitation:** `missing_step` errors are exactly the
non-localizable segments — a step that never happened produces no time window,
so a windowed detector **cannot emit it** (test support = 2 windows, F1 = 0.0 in
every row). In practice the pipeline covers **7 of 8 classes**; macro-F1 must be
reported with and without it.

**Test-set support** (windows): no_mistake 16,584 · order_error 4212 ·
technique_error 846 · measurement_error 786 · preparation_error 621 ·
timing_error 556 · temperature_error 89 · missing_step 2.

---

## 4. Training (Contribution A)

`scripts/cc4d_lora_train.py` → `train/lora_train.py`.

- Qwen2.5-VL-7B-Instruct, LoRA r=32, α=64, dropout 0.05 on q/k/v/o/gate/up/down
  → **95M trainable params (1.13%)**.
- **Completion-only LM cross-entropy** — the model *generates* the target JSON;
  prompt and image tokens are masked. The class set lives in the prompt/targets,
  **not** in a classifier head.
- 10,000 class-balanced windows (`neg_per_pos=1.0`, ~4951+/5049−), stride 4 s,
  1 fps, 2 epochs, lr 2e-4 cosine, grad-accum 8. ~3 h on one RTX PRO 6000.
- Final train loss ≈ 0.017. Adapter: `outputs/lora_cc4d_8class/` (364 MB),
  merged at inference via `merge_and_unload()`.

---

## 5. Results — full 109-clip test ablation (complete)

`outputs/full_eval_8class/ablation.{json,md}`, stride 4 s, all 109 test clips,
23,696 windows, no subsetting.

### Online (causal)

| row | binary F1 | macro F1 | AUC | P | R | ECE | EDS | start-MAE (s) |
|---|---|---|---|---|---|---|---|---|
| zero-shot + fixed | 0.036 | 0.107 | 0.497 | 0.259 | 0.019 | 0.549 | 0.245 | 236.2 |
| zero-shot + adaptive | 0.017 | 0.104 | 0.497 | 0.212 | 0.009 | 0.548 | 0.115 | 317.6 |
| **LoRA + fixed** | **0.471** | **0.185** | **0.616** | 0.408 | 0.556 | 0.502 | **0.763** | **60.2** |
| LoRA + adaptive | 0.373 | 0.170 | 0.580 | 0.451 | 0.318 | 0.472 | 0.369 | 102.1 |
| MistSense baseline | 0.495 | — | — | — | — | — | — | — |

Per-class F1, LoRA + fixed (online): order 0.326 · technique 0.167 · timing
0.151 · measurement 0.074 · preparation 0.058 · temperature 0.000 ·
missing_step 0.000 · no_mistake 0.705.
Localization: 55/61 localizable events detected (adaptive: 45/61).

### Offline (full-video)

Same ordering, uniformly lower: zero-shot 0.040/0.022, LoRA **0.445** (fixed) vs
0.297 (adaptive). The gap is explained by the saturating step conditioning
(§2).

### What the table says

1. **Contribution A is real and large.** Graph-conditioned LoRA fine-tuning
   moves binary F1 from 0.036 → 0.471 online (0.040 → 0.445 offline) on
   identical clips, windows, and protocol — a like-for-like comparison, unlike
   any earlier subset number. AUC 0.497 → 0.616. Zero-shot Qwen2.5-VL is
   essentially a non-detector here (recall 0.019).
2. **Contribution B is currently falsified.** Adaptive τ loses to a fixed
   τ=0.5 in **all four** comparisons — LoRA online −0.098 F1, LoRA offline
   −0.148, and both zero-shot rows. It trades a large recall loss (0.556 →
   0.318) for a small precision gain (0.408 → 0.451), and it halves earliness
   (EDS 0.763 → 0.369). The LoRA fit was genuine (`fitted: true`,
   `w_conf = +1.447`); the zero-shot fit collapsed to defaults
   (`nonpositive_confidence_coef`, `w_conf = −0.057`).
3. **The fitted weights contradict the design intuition.** α = **−0.405**
   (negative): the data says *more complex tasks get a lower bar*, the opposite
   of the stated design. Full fit: α −0.405, β 0.447, γ 0.868, δ 0.089,
   ε 0.525, bias 0.481.
4. **The baseline still edges us on binary F1.** MistSense 0.495 vs our best
   0.471 — but MistSense is **binary-only** (no typing), runs at 2 fps with a
   different window definition, and its τ was chosen by a post-hoc oracle
   sweep. Not directly comparable; a matched re-run is still owed.
5. **Calibration is poor across the board** (ECE ≈ 0.47–0.55, Brier ≈ 0.49–0.52).
   The token-probability confidence is *informative* (it unblocked the fit and
   carries the correct polarity) but it is **not calibrated**.
6. **Macro-F1 is low (0.185)** and dominated by near-zero rare-class scores;
   report it alongside support, with and without `missing_step`.

---

## 6. Repo map

```
src/vmd/
  data/      mistake_classes.py (taxonomy) · captaincook4d.py (loader)
             captaincook4d_graphs.py (gold DAG → ProceduralGraph) · sampler.py
             egoops.py (dataclasses only — names kept for continuity)
  graph/     schema.py (pydantic ProceduralGraph) · builder.py (LLM/GPT-4o) · validator.py
  vlm/       qwen25vl.py (model + prompt + token-prob confidence) · uncertainty.py · lora.py
  decision/  threshold.py (AdaptiveThreshold + fit) · postprocess.py
  eval/      online.py · offline.py · metrics.py · report.py
  train/     lora_train.py · losses.py
  cli.py     `vmd build-graph | infer | eval | train`

scripts/     cc4d_full_eval.sh (4-pass orchestrator) · cc4d_zeroshot_eval.py (zero-shot
             AND LoRA eval driver) · cc4d_lora_train.py · cc4d_fit_and_eval_threshold.py
             · cc4d_build_ablation.py · cc4d_phase9_chain.sh · 03b_fit_threshold.py
configs/     base.yaml · cc4d.yaml · qwen25vl_zeroshot.yaml · qwen25vl_lora.yaml
outputs/     lora_cc4d_8class/ (current adapter) · full_eval_8class/ (headline results)
             + 6-class backups, subset runs, loss curve, pipeline diagrams
baseline/    MistSense/ (comparison baseline, own train/eval)
tests/       48 unit tests, no GPU or data required
```

**Commits:** `ba47567` snapshot · `208bb5c` drop EgoOops + 8-class taxonomy
(26 files, +196/−847) · `f32fd4b` harden eval harness (9 files, +518/−36) ·
`141385c` token-probability confidence (2 files, +121/−9).

**Uncommitted:** `scripts/cc4d_full_eval.sh` modified, `_smoke_token_conf.py`
deleted, and the four report markdowns are untracked.

---

## 7. Reproducing

```bash
pip install -e .
pytest tests/ -v                                  # 48 tests, no GPU/data

# Contribution A
python scripts/cc4d_lora_train.py --output_dir outputs/lora_cc4d_8class

# Full ablation: LoRA-val→fit τ, zs-val→fit τ, zs-test, LoRA-test, aggregate
bash scripts/cc4d_full_eval.sh                    # ~38 h on one GPU
# → outputs/full_eval_8class/ablation.{json,md}
```

Env: `OPENAI_API_KEY` (LLM graph building only — the gold CC4D graphs are used
for all reported results), `VMD_ROOT`, `WANDB_API_KEY`.

---

## 8. Open problems, in priority order

1. **Decide Contribution B's fate.** It is falsified as stated. Options: (a)
   reframe it as *calibration* of token-probability confidence (temperature /
   Platt / conformal) rather than an adaptive bar; (b) per-class thresholds;
   (c) report it as a negative result with the α-sign analysis. Do not ship the
   current framing.
2. **Explain the negative α**, or drop/re-sign `task_complexity`
   (`len(graph.steps)/20`) — it may simply be a proxy for clip length.
3. **Statistical layer — still missing.** Every number above is a single-run
   point estimate, one seed, no CIs. Need per-clip bootstrap CIs and paired
   tests (McNemar on window decisions) for zero-shot-vs-LoRA (certainly
   significant) and fixed-vs-adaptive (certainly significant, wrong direction).
4. **Matched MistSense re-run** at 1 fps / stride 4 with a non-oracle τ, or a
   loud footnote — currently the baseline beats us on the only metric it
   reports.
5. **`graph_confidence` is hardcoded 0.8** in both eval loops — the δ feature
   is constant, so it contributes nothing but a bias offset. Either make it
   real (graph-match score) or remove the term.
6. **Missing ablations:** graph-conditioned vs unconditioned prompt (this is the
   load-bearing claim of Contribution A and is *not yet isolated* — the current
   table conflates fine-tuning with graph conditioning), and gold vs LLM-built
   graph.
7. **Stale docs:** `CLAUDE.md`, `RESULTS_SUMMARY.md`, `pipeline_diagram.md`, and
   `PROGRESS_REPORT.md` §9–§10 still describe EgoOops / 6 classes / a running
   eval.

---

## 9. One paragraph

The pipeline slices CaptainCook4D videos into 4 s causal windows, conditions
Qwen2.5-VL-7B on the recipe's gold procedural graph, reads a genuine confidence
from the model's `deviation` decision-token probability, gates on a threshold,
smooths, and scores against 8-class annotations both online and offline. On the
complete 109-clip test ablation, graph-conditioned LoRA fine-tuning lifts binary
F1 from 0.036 to 0.471 online — Contribution A is demonstrated end-to-end — while
the adaptive threshold loses to a fixed one everywhere, so Contribution B needs
reframing or retirement; and the remaining gaps are statistical validity, a
matched baseline, and an ablation that isolates graph conditioning from
fine-tuning.
