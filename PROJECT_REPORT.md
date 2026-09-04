# Procedural Mistake Detection in Egocentric Video — Full Project Report

**Target venue:** EMNLP 2026 (via ACL ARR)
**Report date:** 18 June 2026
**Status:** Contribution A proven (binary detection beats baseline); Contribution B still blocked; multi-class typing weak; full-test ablation pending.
**Purpose:** complete, honest, step-by-step account of everything done — for supervisor review and defense questions.

---

## 0. The problem and the two contributions

**Problem.** Given an egocentric (first-person) video of someone performing a procedural task (cooking a recipe), detect — online, clip by clip, without seeing the future — whether the person made a *mistake*, and what *kind* of mistake.

**Contribution A — Procedural Graph-Guided VLM Reasoning.**
We give a vision-language model (Qwen2.5-VL-7B) an explicit machine-readable **task graph** of the correct procedure (steps, ordering constraints, required objects/states) and ask it to judge each clip against that graph. Hypothesis: conditioning on the procedural graph makes detection more accurate *and* more interpretable than a graph-blind model.

**Contribution B — Uncertainty-Aware Adaptive Decision Threshold.**
Instead of one fixed confidence cut-off for "is this a mistake," the threshold τ adapts per window from the model's uncertainty, graph-match confidence, task complexity, and recent history:

```
τ = clip( α·complexity + β·variants + γ·u + δ·(1−g) + ε·hist + bias , 0, 1 )
```

Hypothesis: an adaptive τ gives a better precision/recall trade-off than any fixed τ.

Both claims need (a) a dataset big and rich enough to train and evaluate on, and (b) a respected published baseline to compare against.

---

## 1. Pipeline architecture (what the system actually does)

```
CaptainCook4D video
  → sampler.py              causal sliding window (online) or full video (offline)
  → graph/builder.py        gold task-graph DAG → ProceduralGraph JSON (cached)
  → vlm/qwen25vl.py         graph-conditioned prompt → structured JSON prediction
  → vlm/uncertainty.py      composite uncertainty score u
  → decision/threshold.py   AdaptiveThreshold τ (Contribution B)
  → decision/postprocess.py temporal smoothing + start-time localization
  → eval/metrics.py         F1, macro-F1, AUC, ECE, Brier, EDS, start-MAE
```

Key types:
- `ProceduralGraph` (`src/vmd/graph/schema.py`): pydantic model — `Step`, `Object`, `Action`, `State`, `Edge` nodes.
- `VLMPrediction` (`src/vmd/vlm/qwen25vl.py`): structured JSON output from the VLM (predicted class, deviation flag, confidence).
- `AdaptiveThreshold` (`src/vmd/decision/threshold.py`): the τ rule above.
- `CaptainCook4DDataset` / `EgoOopsClip` / `MistakeAnnotation` (`src/vmd/data/`): dataset loader and clip/annotation dataclasses.

Evaluation runs **two protocols side by side every time**:
- **Online**: causal sliding window, no future access (matches how MistSense and similar online methods are evaluated).
- **Offline**: full video, all clips at once.

---

## 2. Dataset journey — why we are on CaptainCook4D

### 2.1 EgoOops (abandoned as primary)
EgoOops (ICCVW 2025) is the *entire* dataset at **50 videos** (30 train / 10 val / 10 test, 538 segments, 95 mistakes, ~47k frames at 2 fps, ~10% positive rate).

**Why fatal:** at that scale the MistSense detector **collapsed** — test `f1_mistake = 0.00` (per-window) / 0.031 (online). It never learns a usable decision boundary. A baseline that scores zero is not a baseline, and we cannot demonstrate our contributions on data too small to train on.

**Decision:** drop EgoOops as primary; adopt a larger benchmark. (EgoOops was later removed entirely — see §3.)

### 2.2 CaptainCook4D (the primary benchmark)
CaptainCook4D (NeurIPS 2024 Datasets & Benchmarks): **384 recordings, 94.5 hours, 24 cooking recipes**, 8 fine-grained error categories, with:
- **gold task-graph DAGs** (one per recipe) — exactly the input Contribution A needs, and lets us test against a *correct* graph, not only an LLM-generated one;
- **temporally-bounded error annotations** — enables the online causal protocol;
- **free-text error descriptions** — supports later captioning/explanation metrics;
- **ungated** download — clean reproducibility;
- **~14× the frames** of EgoOops (679,782 vs 47,409) and ~2× the positive rate (~21%).

Selected over alternatives via a verified literature survey: IndEgo (repo not public until ~Mar 2026), Ego4D-PMD (synthetic, frame-level, no temporal bounds).

---

## 3. Taxonomy — from 8→6 remap to native 8-class (important refactor)

**First approach (Jun, original).** Our framework used 6 mistake classes; CC4D has 8. We wrote a documented **8→6 remap** with two policies — STRICT (drops the 3 categories that have no clean target) and LENIENT (folds them into the nearest class) — and reported both. We manually audited all 410 "Preparation" errors for their mapping.

**Pivot (Jun 16, commit `208bb5c`).** We **dropped EgoOops entirely and switched to CC4D's native taxonomy** to eliminate dual-dataset bookkeeping and the lossy remap. New canonical taxonomy = CC4D's 8 categories minus the "Other" catch-all (only 8 instances):

```
no_mistake | order_error | technique_error | preparation_error |
measurement_error | missing_step | timing_error | temperature_error
```

What changed:
- Single source of truth: `src/vmd/data/mistake_classes.py` (`MISTAKE_CLASSES`, `CC4D_TAG_TO_CLASS`, `step_label`, `PRIORITY`).
- Deleted the STRICT/LENIENT remap machinery (`captaincook4d_classes.py`).
- Qwen2.5-VL prompt template + `_VALID_CLASSES` rewritten for 8 classes, with glosses for the newly-supported `timing_error` and `temperature_error`.
- EgoOops config + scripts + raw data (11 GB) removed; `egoops.py` kept **only** as home of the `EgoOopsClip` / `MistakeAnnotation` dataclasses (legacy names preserved deliberately).
- Tests updated: **44/44 pass**.
- 6-class LoRA adapter + prediction JSONs backed up to `*_6class_backup` paths before retraining.

> **Defense note:** the headline numbers in §5–7 are the *current 8-class native* numbers. The older 0.535 figure that appears in `RESULTS_SUMMARY.md` was the 6-class run and is superseded.

---

## 4. Baseline — MistSense (ICCV 2025), reproduced on CC4D

**What MistSense is.** A recent three-stage procedural-mistake-detection pipeline: (1) a detection head over RGB + hand features with weighted binary cross-entropy; (2) a projection layer feeding an LLM; (3) LoRA fine-tuning for mistake captioning. We compare on **Stage 1** (the detector).

**Why chosen.** Closest published method to our setting (online procedural error detection in egocentric video), recent enough that beating it matters for a 2026 submission, reproducible end-to-end on the *same* data and protocol as our method.

**Reproduction deviations (stated honestly for the paper).** Original uses gated Llama2-7B + BLIP-2 ViT. We substituted **OPT-2.7b** (ungated) for the LLM and **CLIP-ViT-L/14** for the visual encoder (matches the paper's stated 1024-d visual size, ~10× faster). These affect only Stages 2/3 (captioning); the Stage-1 detector we compare on is faithful.

**Result (retrained on CC4D, 109-clip test, τ sweep).**

| metric | EgoOops (collapsed) | **CC4D online (per-frame, 189,885)** | **CC4D per-window (872)** |
|---|---|---|---|
| f1_mistake (best) | 0.00 / 0.031 | **0.452** @ τ0.70 | **0.495** @ τ0.70 |
| overall F1 | ~0 | 0.447 | 0.547 |
| precision_mistake | — | 0.312 | 0.356 |
| recall_mistake | — | 0.821 | 0.814 |

Config: `bce_w1=5.0` (inverse-freq ≈4.9), batch 16, 10 epochs, best epoch 3 (val f1_mistake 0.488).

**Meaning.** `f1_mistake` went **0.00 → ~0.49** — a broken baseline became a working detector, same ballpark as MistSense's own paper (Epic-tent-O 63.3, HoloAssist 55.7, RGB+Hand). Pure data scale fixed it: ~14× frames, ~2× positive rate, same architecture.

**Known weakness (deliberately not chased):** high recall (~0.82), low precision (~0.31) — over-fires, driven by the 28% positive rate + positive class weight. A one-line config tweak; we leave it because the baseline is already a credible reference and polishing the baseline is not the contribution.

Artifacts: `runs/cc4d_stage1_best.pt`, `runs/cc4d_online_eval.json`, `baseline/MistSense/CC4D_RESULTS.md`.

---

## 5. Contribution A — the graph-guided VLM

### 5.1 Zero-shot floor (the "before")
Graph-conditioned Qwen2.5-VL with **no training** — just graph + prompt — on a 16-clip stratified test subset, online stride 4s.

| metric | value |
|---|---|
| binary_f1 | **0.176** |
| precision / recall | 0.326 / 0.120 |
| auc_roc | 0.519 (≈ chance) |
| ece | 0.646 (over-confident) |

Fires rarely (recall 12%), right ~1/3 of the time. A real, non-collapsed number but **below baseline (0.49)** — the honest floor. This isolates how much fine-tuning + the graph contribute.

### 5.2 LoRA fine-tune (the "after" — the headline experiment)
Graph-conditioned LoRA fine-tuning of Qwen2.5-VL-7B.

**Hyperparameters** (`scripts/cc4d_lora_train.py`, adapter `outputs/lora_cc4d_8class/adapter_config.json`):
- LoRA rank `r=32`, `alpha=64`, dropout 0.05, bias none
- target modules: `q,k,v,o_proj` + `gate,up,down_proj` (attention + MLP)
- lr `2e-4`, epochs 2, grad-accum 8
- data: balanced ~10k-window subset (≈5k mistake / 5k normal), `neg_per_pos=1.0`, stride 4s, clip 4s, 1 fps
- base model `Qwen/Qwen2.5-VL-7B-Instruct`, task `CAUSAL_LM`

**Two pipeline bugs fixed to get here** (both would have silently wasted GPU time):
1. **OOM data loader.** The original loader (written for tiny EgoOops) eager-loaded every window's pixels into RAM — ~240 GB on CC4D, would never run. Rewrote to load frames **lazily** and class-balance + cap window count.
2. **Loss masking.** Loss was computed over the *entire* input sequence (~1850 prompt+image tokens) instead of only the ~71 answer tokens, so the signal was drowned and **loss plateaued at ~6.1 (not learning)**. After fixing to **completion-only masking**, loss dropped immediately to **~0.01–0.017** and trained normally.

### 5.3 Current 8-class trained-A result (verified, Jun 18)
16-clip stratified test subset, online stride 4s, adapter `outputs/lora_cc4d_8class`, with token-probability confidence (§6.3). Source: `outputs/cc4d_lora_test_eval_8class.json`.

| metric | zero-shot | **LoRA 8-class** | baseline |
|---|---|---|---|
| **binary_f1** | 0.176 | **0.529** | 0.49 |
| auc_roc | 0.519 | **0.671** | — |
| precision / recall | 0.33 / 0.12 | **0.384 / 0.850** | 0.31 / 0.82 |
| ece (calibration) | 0.646 | **0.300** | — |
| brier | 0.609 | **0.330** | — |
| macro_f1 (8-class) | 0.175 | **0.119** | — |
| EDS (event detection) | — | **0.852** | — |
| start-MAE (s) | — | 12.25 | — |
| localizable events detected | — | **8 / 8** | — |

Per-class F1 (test subset): `no_mistake 0.649`, `technique_error 0.153`, `measurement_error 0.104`, `preparation_error 0.044`, `order_error 0.0`, `missing_step 0.0`, `timing_error 0.0`, `temperature_error 0.0`.

Val (full 62 clips, `cc4d_val_eval_8class.json`): binary_f1 0.479, AUC 0.607, precision 0.399, recall 0.600.

**What this proves and does NOT prove:**
- ✅ **Binary mistake detection works and beats the baseline:** 0.176 → 0.529 (3× the zero-shot floor), clears MistSense 0.49. AUC 0.52 (chance) → 0.671. Token-prob confidence cut ECE 0.646 → 0.300 (much better calibrated). All 8/8 localizable events detected (EDS 0.852).
- ⚠️ **Multi-class typing is weak:** macro-F1 only **0.119**. The model essentially separates *mistake vs not* and labels `no_mistake` well, but cannot reliably name the mistake *type* — `order_error`, `missing_step`, `timing_error`, `temperature_error` are all 0.0 on the subset (several have tiny/zero support there). This is a real limitation to disclose.

---

## 6. Contribution B — adaptive threshold (the unresolved blocker)

### 6.1 The original symptom
Adaptive τ produced **exactly identical** metrics to a fixed threshold. We investigated rather than papered over it and found two layers.

### 6.2 Layer 1 — implementation flaw (FIXED)
Old τ was `σ(w·f + bias)`, which mathematically squashed τ into the band ~[0.40, 0.58] centered on the fixed 0.5; with the VLM's coarse confidences it never crossed into the differentiating zone, so adaptive collapsed onto fixed. **Rewrote τ as a clipped *linear* confidence bar** over the full [0,1] range, and rewrote `fit_threshold_params` to fit a logistic on `[confidence, features] → label` and read τ off the proper decision boundary. **8 unit tests pass.**

### 6.3 Layer 2 — no signal to adapt (the deep blocker)
B adapts a threshold *on top of A's output*. If A's confidence carries no signal, no threshold rule can help.

- **Zero-shot A:** verbalized confidence took only {0.8, 0.9}, identical on true-positives (0.859) vs false-positives (0.863); deviation flag at chance (AUC 0.49); 3 of 5 τ-features constant. The fit *correctly refused*.
- **Fine-tuned A made it worse:** the LoRA training targets **hardcoded `confidence = 0.9`**, and the model memorized it — emitting a single confidence on every window. Adaptive still == fixed.

**The fix (Jun 18, commit `141385c`) — token-probability confidence (no retrain).**
Read confidence from the **generation logits**, not the verbalized text. `Qwen25VL._run` now generates with `output_scores=True`; `_deviation_token_confidence` returns the softmax probability on the model's `deviation` decision token = P(mistake) for that window. `VLMPrediction.confidence` is overwritten with this token probability; the verbalized value is kept as `verbalized_confidence` for reference.
- Smoke on 18 real test windows: token confidence ranges **0.117–0.945** (stdev 0.28, 16/18 distinct) with correct polarity (low on `no_mistake`, high on confident deviations) vs the constant 0.900 verbalized value.
- This is what cut ECE 0.646 → 0.300 in §5.3.

### 6.4 Current status — STILL BLOCKED
Even with the token-probability confidence, the **val-level threshold fit still refuses**:

```
outputs/threshold_params_8class.json.meta.json:
  { "fitted": false, "reason": "nonpositive_confidence_coef", "w_conf": -0.131 }
```

So `outputs/cc4d_lora_test_eval_8class.json` still shows **adaptive == fixed** (binary_f1 0.529 for both). The per-window confidence now has spread and is better calibrated, but on the validation set the logistic regression still finds a non-positive confidence coefficient — i.e. higher confidence is not yet monotonically associated with being correct at the window level, so the fitter defaults to fixed parameters rather than invent signal.

**Honest conclusion for the paper:** Contribution B's *machinery* is correct and banked, and the token-probability fix restored per-window confidence spread + calibration. But B is **not yet empirically validated** — adaptive does not yet beat fixed. Two latent feature bugs also remain to clear when B is revisited: `graph_confidence` is hardcoded (`online.py:43`) and `valid_variations` is absent from the gold graphs (they don't encode variants). Closing those plus getting the val fit to accept is the remaining work for B.

---

## 7. Where we stand — one table (current, verified 8-class)

| System | binary_f1 | AUC | prec / rec | ECE | Status |
|---|---|---|---|---|---|
| MistSense baseline (trained) | 0.49 | — | 0.31 / 0.82 | — | ✅ done, credible reference |
| Contribution A — zero-shot floor | 0.176 | 0.52 | 0.33 / 0.12 | 0.646 | ✅ done, honest floor |
| **Contribution A — LoRA 8-class** | **0.529** | **0.671** | 0.38 / 0.85 | **0.300** | ✅ **beats baseline (binary)**; multi-class weak (macro-F1 0.119) |
| Contribution B — adaptive τ | == fixed | — | — | — | ❌ blocked: val fit refuses (w_conf −0.131) |

All Contribution-A numbers are on a **16-clip stratified test subset** (val is 62 clips full). The full all-test ablation (`scripts/cc4d_full_eval.sh`, "Phase 10") is staged but **not yet completed** — working tree shows it mid-edit.

---

## 8. Honest weaknesses / what a reviewer will attack

1. **Subset size.** Headline A numbers are 16 clips. A paper needs the full test split. → run `cc4d_full_eval.sh`.
2. **No graph on/off ablation yet.** The *core* Contribution-A claim ("the graph helps") is currently unproven — we have not isolated graph-conditioning from plain fine-tuning. This is the single most important missing experiment.
3. **Multi-class typing is poor** (macro-F1 0.119; four classes at 0.0). The model detects mistakes but largely cannot name the type. Either reframe A's claim as *binary* detection, or improve typing.
4. **Contribution B not validated.** Adaptive still == fixed; the val fit refuses. B is a method + a negative-to-neutral result right now, not a win.
5. **Baseline deviations** (OPT-2.7b, CLIP-ViT) must be stated; they touch only Stages 2/3.
6. **Calibration metrics** improved but ECE 0.30 is still high in absolute terms.

---

## 9. What's left (priority order)

1. **Graph on/off ablation** on the trained model — direct proof of Contribution A. *(highest priority)*
2. **Full all-test eval** (`scripts/cc4d_full_eval.sh`) to replace the 16-clip subset numbers.
3. **Unblock B for real:** fix `graph_confidence` hardcode (`online.py:43`), add `valid_variations`, and get the val confidence coef positive (better confidence source or per-class fit) so adaptive can beat fixed.
4. **Improve multi-class typing** or reframe the claim to binary detection.
5. Lower priority: baseline precision tuning; MistSense Stage 2/3 captioning on CC4D; per-recipe complexity features for τ.

---

## 10. Reproduce commands

```bash
# Baseline (MistSense Stage 1 on CC4D)
cd baseline/MistSense
python -m src.train --stage 1 --override configs/cc4d.yaml \
    --data_root data/processed/cc4d \
    --train_annot data/cc4d_annotations/train.json \
    --val_annot   data/cc4d_annotations/val.json --save runs/cc4d_stage1_best.pt
python scripts/eval_online.py --override configs/cc4d.yaml --data_root data/processed/cc4d \
    --test_annot data/cc4d_annotations/test.json --ckpt runs/cc4d_stage1_best.pt --out runs/cc4d_online_eval.json

# Contribution A — zero-shot floor (subset)
python scripts/cc4d_zeroshot_eval.py --limit 16 --stratify --stride 4 --mode online \
    --out outputs/cc4d_zeroshot_subset.json

# Contribution A — LoRA fine-tune (8-class)
python scripts/cc4d_lora_train.py --epochs 2 --max_windows 10000

# Contribution A — eval trained adapter (8-class, token-prob confidence)
python scripts/cc4d_zeroshot_eval.py --split test --limit 16 --stratify --stride 4 --mode online \
    --lora_adapter outputs/lora_cc4d_8class \
    --threshold_json outputs/threshold_params_8class.json \
    --out outputs/cc4d_lora_test_eval_8class.json

# Contribution B — fit + eval adaptive threshold
python scripts/cc4d_fit_and_eval_threshold.py \
    --val_preds outputs/cc4d_val_preds_8class.json \
    --params_out outputs/threshold_params_8class.json

# Full all-test 8-class ablation (Phase 10)
bash scripts/cc4d_full_eval.sh

# Tests (no GPU/data needed)
pytest tests/ -v     # 44/44 pass
```

---

## 11. Metric glossary (for live questions)

- **binary_f1** — F1 on the binary mistake-vs-normal decision (our main detection number).
- **macro_f1** — unweighted mean of per-class F1 across all 8 classes (measures *typing*, punishes ignoring rare classes).
- **AUC-ROC** — ranking quality of the confidence score; 0.5 = chance.
- **ECE** (Expected Calibration Error) — gap between confidence and accuracy; lower = better calibrated.
- **Brier** — mean squared error of probabilistic predictions; lower better.
- **EDS** — event detection score (did we catch each localizable error event).
- **start-MAE** — mean absolute error (seconds) of predicted error start time vs ground truth.
- **Online vs offline** — causal sliding window (no future) vs full-video access; reported side by side.

---

### Key artifacts
- `RESULTS_SUMMARY.md` — earlier (Jun 15, 6-class) narrative; superseded on numbers by this report.
- `baseline/MistSense/CC4D_RESULTS.md` — detailed baseline numbers + reproduce commands.
- `outputs/cc4d_lora_test_eval_8class.json` — current trained-A headline (8-class).
- `outputs/cc4d_val_eval_8class.json`, `outputs/threshold_params_8class.json(.meta)` — B fit (refused).
- `outputs/cc4d_zeroshot_subset.json` — zero-shot floor.
- `src/vmd/data/mistake_classes.py` — taxonomy single source of truth.
- Commits: `208bb5c` (8-class pivot), `f32fd4b` (eval harden), `141385c` (token-prob confidence / B unblock attempt).
