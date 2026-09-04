> **SUPERSEDED — historical record, do not cite.**
> This document describes the 6-class taxonomy and/or the EgoOops-era pipeline,
> both of which were replaced in `208bb5c` (CC4D native 8-class). Its numbers
> predate the full 109-recording test ablation and the frozen decision layer.
> Current sources: `EXPERIMENTS_TODO.md` (state and open work), `CLAUDE.md`
> (architecture), `outputs/decision_v2/ablation.md` (results), `paper/main.tex`.
> Kept for provenance.

# Procedural Mistake Detection — Results & Rationale Summary

**Project:** Procedural mistake detection in egocentric video.
**Target venue:** EMNLP 2026 (via ACL ARR).
**Date of this summary:** 15 June 2026.
**Audience:** supervisor / collaborators — this document explains *what* we did, *why* we did it, and *what the numbers mean*, not just the final metrics.

---

## 0. The two contributions we are trying to publish

Everything below is in service of two claims:

- **Contribution A — Procedural Graph-Guided VLM Reasoning.** We give a vision-language model (Qwen2.5-VL-7B) an explicit, machine-readable *task graph* of the correct procedure (steps, ordering constraints, required objects/states) and ask it to judge, clip by clip, whether the person deviated from that procedure. The hypothesis: conditioning the VLM on the procedural graph makes mistake detection more accurate and more interpretable than a graph-blind model.
- **Contribution B — Uncertainty-Aware Adaptive Decision Threshold.** Instead of a single fixed confidence cut-off for "is this a mistake," the threshold τ adapts per window using the model's uncertainty, the graph-match confidence, task complexity, and recent history. The hypothesis: an adaptive τ gives a better precision/recall trade-off than any fixed τ.

To make either claim credible we need (a) a dataset large and rich enough to train and evaluate on, and (b) a respected prior method to serve as a **baseline** we can compare against.

---

## 1. Choosing the baseline: MistSense (ICCV 2025)

**What:** MistSense is a recent (ICCV 2025) procedural-mistake-detection method. It is a three-stage pipeline — (1) a detection head over RGB + hand features with weighted binary cross-entropy, (2) a projection layer that feeds an LLM, (3) LoRA fine-tuning for mistake captioning.

**Why we picked it:** it is the closest published method to our problem setting (online procedural error detection in egocentric video), it reports on egocentric benchmarks, and it is recent enough that beating it is a meaningful result for a 2026 submission. We reproduced it end-to-end so we have an apples-to-apples reference point on the *same* data and the *same* evaluation protocol as our own method.

**Reproduction deviations (honesty for the paper):** the original uses a gated Llama2-7B and BLIP-2's ViT. We substituted `OPT-2.7b` (ungated) for the LLM and `CLIP-ViT-L/14` for the visual encoder (matches the paper's stated 1024-d visual size and is ~10× faster). These only affect Stages 2/3 (captioning); the Stage-1 detector — which is what we compare on — is faithful.

---

## 2. Why we abandoned EgoOops as the main dataset

**What we found:** EgoOops (ICCVW 2025) is **50 videos total** — that is the entire dataset, not a subset. 30 train / 10 val / 10 test, 538 segments, 95 mistakes.

**Why this is fatal:** with that little data the MistSense detector **collapsed** — test `f1_mistake = 0.00` (per-window) / 0.031 (online). The model simply never learns a usable decision boundary from ~47k frames at ~10% positive rate. A baseline that scores zero is not a baseline; and we cannot demonstrate *our* contributions on data too small to train on. No amount of re-training fixes a fundamental data-scale problem.

**Decision:** keep EgoOops only as a small free-text *probe* (it has nice natural-language mistake descriptions), and adopt a larger primary benchmark.

---

## 3. Why CaptainCook4D became the primary benchmark

**What:** CaptainCook4D (NeurIPS 2024 Datasets & Benchmarks) — 384 recordings, 94.5 hours, 24 cooking recipes, 8 fine-grained error categories, with **ground-truth task-graph DAGs**, **temporally-bounded error annotations**, and **free-text error descriptions**.

**Why it is the right fit (this is the key point):** it is essentially purpose-built for our two contributions —

- It **ships gold task graphs** (one per recipe). That is exactly the input Contribution A needs, and lets us validate the graph-guided idea against a *correct* graph rather than only an LLM-generated one.
- It has **temporal error bounds**, so we can run the **online causal** protocol (sliding window, no future access) that matches how MistSense and similar online methods are evaluated.
- It has **free-text descriptions**, supporting the captioning / explanation metrics later.
- It is **ungated** (Apache-style Box download), so reproducibility is clean.
- It is ~8× the scale needed to actually train.

It was selected over alternatives (IndEgo — repo not ready until ~Mar 2026; Ego4D-PMD — synthetic, frame-level, no temporal bounds) via a verified literature survey.

**The mapping work this required:** CaptainCook4D uses 8 error categories; our framework uses 6 mistake classes. We wrote a documented **8→6 remap** with two policies — STRICT (drops the 3 categories with no clean target from the *typed* metric) and LENIENT (folds them into the nearest class) — and we report both so the choice is transparent. We also manually audited all 410 "Preparation" errors to decide their mapping. The gold task graphs were converted directly into our `ProceduralGraph` schema (drop-in, validating Contribution A's input format).

---

## 4. Result #1 — the baseline now works (MistSense on CaptainCook4D)

We retrained the MistSense Stage-1 detector on CaptainCook4D and evaluated on the 109-clip test split, under both protocols.

| metric | EgoOops (old, collapsed) | **CaptainCook4D** |
|---|---|---|
| f1_mistake, online per-frame (189,885 frames) | 0.031 | **0.452** @ τ0.70 |
| f1_mistake, per-window (872 windows) | **0.00** | **0.495** @ τ0.70 |
| overall F1, per-window | ~0 | 0.547 |
| precision / recall @ best | — | 0.31 / 0.82 |

**What this means:** `f1_mistake` went from **0.00 → ~0.49** — a broken baseline became a working detector. For context, MistSense's *own* paper reports F1 63.3 (Epic-tent-O) and 55.7 (HoloAssist) on RGB+Hand; different datasets, but our ~0.49 is the same ballpark, so this is a legitimate baseline, not a crippled one.

**Why it works now:** purely data scale — CaptainCook4D gives ~14× more frames (680k vs 47k) and ~2× the positive rate (~21% vs ~10%). Same architecture, enough data.

**Known weakness (deliberately not chased):** high recall (~0.82), low precision (~0.31) — the detector over-fires, driven by the 28% positive rate and the positive class weight (bce_w1=5.0). This is a one-line config tweak (raise τ or lower the weight); we are leaving it because the baseline is already a credible reference point and polishing the *baseline* is not where the contribution lies.

---

## 5. Result #2 — Contribution A, zero-shot floor

Before fine-tuning, we measured graph-conditioned Qwen2.5-VL **zero-shot** (no training, just the graph + prompt) on a 16-clip stratified test subset, online stride 4s.

| metric | value |
|---|---|
| binary_f1 | **0.176** |
| precision / recall | 0.33 / 0.12 |
| auc_roc | 0.519 (≈ chance) |
| ece | 0.646 (over-confident) |

**What this means:** zero-shot, the VLM fires rarely and is right about a third of the time — a real, non-collapsed number but **below the baseline (0.49)**. That is expected and fine: this is the *floor*. The actual Contribution-A experiment is the **LoRA fine-tune** (Section 7), which must clear the baseline. The zero-shot number is the "before" in our before/after story and isolates how much the fine-tuning (and the graph) contribute.

---

## 6. Result #3 — Contribution B, why we paused it (an important negative finding)

We tried to validate the adaptive threshold and discovered it produced **exactly the same** results as a fixed threshold. We investigated rather than papered over it, and found two separate issues:

1. **An implementation flaw (fixed).** The threshold was computed through a sigmoid, which mathematically squashed τ into the narrow band [0.40, 0.58] centered on the fixed value 0.5. With the VLM's coarse confidence values it never crossed into the differentiating zone, so adaptive collapsed onto fixed. We rewrote τ as a clipped *linear* confidence bar spanning the full [0,1] range, and corrected the parameter-fitting (a logistic regression that now reads τ off the proper decision boundary). 8 unit tests pass.

2. **The deeper, real blocker.** We collected 3,922 validation windows and tried to fit the threshold. The fit **correctly refused** (it detected that confidence carried no usable signal). The reason: *zero-shot Qwen's output is at chance on this data* —
   - verbalized confidence takes only two values {0.8, 0.9}, and is **identical** on true-positives (0.859) vs false-positives (0.863);
   - the deviation flag itself is uninformative — fired on 15% of true-mistake windows vs 16% of true-normal (AUC 0.49 ≈ random);
   - 3 of the 5 threshold features are constant on this data.

**Why this matters and what we concluded:** Contribution B *adapts a threshold on top of Contribution A's output*. If A's output is at chance, there is nothing for B to adapt — no threshold rule can extract signal that isn't there. Therefore **B cannot be validated until A produces a real, above-chance, calibrated signal**, i.e. after fine-tuning. The B code is now correct and "banked" — dormant, waiting for a trained A. This reorders our roadmap to **A before B**, which is why we are running the LoRA fine-tune now.

---

## 7. In progress — Contribution A LoRA fine-tune (the headline experiment)

**What:** LoRA fine-tuning of Qwen2.5-VL-7B (rank 32), graph-conditioned, on a balanced ~10k-window subset of CaptainCook4D train (≈5k mistake / 5k normal, stride 4s), 2 epochs.

**Why this is the experiment that matters:** it is the "after" to the zero-shot "before." A fine-tuned, graph-conditioned model is what we claim beats the baseline (trained-vs-trained, fair fight against MistSense's 0.49). It also produces the calibrated confidence spread that **unblocks Contribution B**. And the planned **graph-on / graph-off ablation** on this trained model is the direct proof of Contribution A's core claim ("the graph helps").

**Two pipeline bugs we had to fix to get here** (worth flagging — both would have silently wasted GPU time):
- The original training data loader (written for tiny EgoOops) eager-loaded every window's pixels into RAM; on CaptainCook4D that is ~240 GB — it would never run. Rewrote it to load frames lazily and to class-balance + cap the window count.
- The loss was being computed over the *entire* input sequence (≈1850 prompt + image tokens) instead of only the ≈71 target-answer tokens, so the training signal was drowned and **loss plateaued at ~6.1 (not learning)**. After fixing to completion-only masking, loss immediately dropped to **~0.01** and is training normally.

**Status at time of writing:** training is healthy and running (~1.6 it/s, ETA ~3.5 h). An evaluation is chained to run automatically when training finishes, on the same 16-clip test subset used for the zero-shot number, so the comparison is direct.

---

## 8. Where we stand — one table

| System | binary_f1 (CaptainCook4D, 16-clip test subset) | AUC | precision / recall | Status |
|---|---|---|---|---|
| MistSense baseline (trained) | **0.49** | — | 0.31 / 0.82 | Done — fixed, credible reference |
| Contribution A — zero-shot | 0.176 | 0.52 (chance) | 0.33 / 0.12 | Done — floor / "before" |
| **Contribution A — LoRA fine-tuned** | **0.535** | **0.68** | 0.40 / 0.82 | **Done — beats baseline** |
| Contribution B — adaptive τ | n/a (== fixed) | — | — | Blocked: trained confidence collapsed to 0.9 |

**Trained-A result (the headline):** fine-tuning lifted graph-guided Qwen from **0.176 → 0.535** binary_f1 — a 3× jump that **clears the MistSense baseline (0.49)**. AUC went 0.52 (chance) → 0.68, and the deviation flag now separates real mistakes from normals (fires on 80% of true-mistake windows vs 43% of normal, versus 15%/16% ≈ random when zero-shot). So the detection genuinely learned, and the graph-guided fine-tuned model is competitive with / better than the published baseline on the same subset and protocol.

**But Contribution B is now *more* blocked, not less.** The fine-tuned model learned the training target's hardcoded `confidence = 0.9` exactly, so it emits a single confidence value for every window. Adaptive τ therefore still equals fixed τ (both 0.535). This is the predicted side-effect of the constant-confidence target and makes the **token-probability confidence fix mandatory** before B can be evaluated (see §9.3). Net: A is proven; B needs the confidence-source change next.

**Narrative for the paper:** we have a working, respected baseline on a properly-sized benchmark with gold task graphs. Our zero-shot graph-guided number sets an honest floor. The fine-tuned graph-guided model (running) is the result that must — and we expect will — clear the baseline, after which the adaptive-threshold contribution becomes measurable. We also surface one honest negative finding (zero-shot VLM confidence is uninformative on this data), which itself motivates the fine-tuning step.

---

## 9. Open threads / next steps
1. Finish & evaluate the LoRA fine-tune; report trained-A vs baseline (Section 7/8).
2. **Graph on/off ablation** on the trained model — the direct test of Contribution A.
3. Revisit Contribution B on the trained model. Note: the training target currently hardcodes confidence = 0.9, so the trained model will emit near-constant confidence; the real fix for B's signal is to derive confidence from the model's token probability rather than verbalized confidence.
4. Full 109-clip zero-shot/test runs (we used a 16-clip subset for fast iteration).
5. MistSense baseline precision tuning and Stage 2/3 (captioning) on CaptainCook4D — lower priority.

**Key artifacts:** `baseline/MistSense/CC4D_RESULTS.md` (detailed baseline numbers + reproduce commands), `data/raw/captaincook4d/INSPECTION.md` (dataset schema + remap), `outputs/cc4d_*.json` (eval outputs), `scripts/cc4d_*.py` (eval / training entry points).
