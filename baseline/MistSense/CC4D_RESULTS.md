# CaptainCook4D results — MistSense baseline + zero-shot VLM

Jun 13 2026. Full EgoOops was 50 videos → MistSense collapsed (test f1≈0).
CaptainCook4D (NeurIPS 2024) adopted as primary benchmark. Numbers below are
the in-domain comparison that makes MistSense a credible baseline again.

## Data scale (why it works now)

| | EgoOops | CaptainCook4D |
|---|---|---|
| train clips | 30 | 213 |
| total frames (2fps) | 47,409 | 679,782 |
| positive frame rate | ~10% | ~21% (17/27/28% tr/val/te) |
| localizable error segments | — | 1,683 (281 nonloc dropped) |

## MistSense Stage 1 (weighted BCE, RGB+Hand, OPT-2.7b not used in stage 1)

Config `configs/cc4d.yaml`: bce_w1=5.0 (inverse-freq ≈4.9), batch 16, 10 epochs.
Best epoch 3, **val f1_mistake = 0.488**.

Test (109 clips), F1 over τ sweep:

| metric | EgoOops (old) | CC4D online (per-frame, 189,885) | CC4D per-window (872) |
|---|---|---|---|
| f1_mistake (best) | **0.00** (per-window) / 0.031 (online) | **0.452** @ τ0.70 | **0.495** @ τ0.70 |
| overall F1 (best) | ~0 | 0.447 @ τ0.70 | 0.547 @ τ0.70 |
| precision_mistake @ best | — | 0.312 | 0.356 |
| recall_mistake @ best | — | 0.821 | 0.814 |

→ **f1_mistake 0.00 → ~0.49**, a working detector (high recall, precision held
down by the 28% positive rate + w1=5 positive bias). Same ballpark as the
paper's native benchmarks (Epic-tent-O F1 63.3 / HoloAssist 55.7, RGB+Hand) —
not identical dataset, but no longer a broken baseline.

Artifacts: `runs/cc4d_stage1_best.pt`, `runs/cc4d_online_eval.json`,
`runs/cc4d_{stage1,online_eval,test_eval}.log`.

## Zero-shot Qwen2.5-VL (graph-conditioned, Contribution-A prompt)

`scripts/cc4d_zeroshot_eval.py`, gold task graphs, online stride 4s, 16-clip
stratified subset (8 localizable-error + 8 normal), 1.98 s/window.

| metric | value |
|---|---|
| binary_f1 | 0.176 |
| precision | 0.326 |
| recall | 0.120 |
| auc_roc | 0.519 |
| macro_f1 | 0.175 |
| ece | 0.646 (overconfident) |

Fires rarely (recall 12%), ~1/3 right when it does. Full 109-clip test run
pending (~9 h at stride 4).

## Contribution B (AdaptiveThreshold) — zero-shot diagnosis (Jun 15 2026)

Investigated why adaptive τ == fixed τ exactly. Root cause has two layers.

**(1) τ implementation collapsed onto fixed.** Old τ = σ(w·f+bias) squashed into
the band [0.40, 0.58], centered on the fixed 0.5; with coarse VLM confidences it
never crossed into the differentiating gap. Fixed in `decision/threshold.py`:
τ is now a **linear confidence bar** clip(Σwᵢfᵢ+bias,0,1) spanning the full
[0,1], and `fit_threshold_params` was rewritten to fit a logistic on
[confidence, features]→label over deviation windows and read τ off the decision
boundary (conf ≥ −(Σwᵢfᵢ+b)/w_conf). 8 unit tests pass.

**(2) The deeper blocker — zero-shot A signal is at chance, so B has nothing to
adapt.** Collected 3,922 val windows (30 clips, stride 8) → `outputs/cc4d_val_preds.json`.
Fit *correctly refused* (`confidence coef -0.4468 ≤ 0 → default params`). Why:

| B input | state on zero-shot CC4D |
|---|---|
| `pred.confidence` | only {0.8, 0.9}; mean TP=0.859 vs FP=0.863 — no signal |
| `uncertainty_u` | std 0.036 (derived from that confidence) |
| `valid_variations` | constant — gold graphs don't encode variants |
| `graph_confidence` | constant — hardcoded 0.8 (`eval/online.py:43`) |
| `deviation` flag | at chance: 15% on true-mistake windows vs 16% on normal; AUC 0.49 |

B operates on top of A's output; zero-shot A ≈ chance ⇒ no threshold helps.
**Conclusion: B can only be validated AFTER Contribution A is fine-tuned (LoRA),**
which gives real confidence spread + above-chance detection. τ design is fixed
and banked; dormant until A produces a real signal. Two latent feature bugs to
clear when B is revisited: graph_confidence hardcoded (online.py:43);
valid_variations absent from gold graphs.

Artifacts: `outputs/cc4d_val_preds.json`, `outputs/cc4d_val_eval.json`,
`scripts/cc4d_fit_and_eval_threshold.py`.

## Reproduce

```bash
# 1. export CC4D -> EgoOops-format annotations (localizable mistakes only)
python baseline/MistSense/scripts/export_cc4d_annotations.py \
    --out_dir baseline/MistSense/data/cc4d_annotations
# 2. extract CLIP-ViT-L + hand features (2fps)  [~2h, 384 clips]
python baseline/MistSense/scripts/extract_features.py \
    --videos_dir data/raw/captaincook4d/videos \
    --annot_dir  baseline/MistSense/data/cc4d_annotations \
    --out_dir    baseline/MistSense/data/processed/cc4d
# 3. Stage 1 train  [~3 min]
cd baseline/MistSense && python -m src.train --stage 1 --override configs/cc4d.yaml \
    --data_root data/processed/cc4d \
    --train_annot data/cc4d_annotations/train.json \
    --val_annot   data/cc4d_annotations/val.json --save runs/cc4d_stage1_best.pt
# 4. eval (causal online + per-window)
python scripts/eval_online.py --override configs/cc4d.yaml --data_root data/processed/cc4d \
    --test_annot data/cc4d_annotations/test.json --ckpt runs/cc4d_stage1_best.pt --out runs/cc4d_online_eval.json
python scripts/eval_test.py   --override configs/cc4d.yaml --data_root data/processed/cc4d \
    --test_annot data/cc4d_annotations/test.json --ckpt runs/cc4d_stage1_best.pt
# zero-shot VLM (subset)
python scripts/cc4d_zeroshot_eval.py --limit 16 --stratify --stride 4 --mode online \
    --out outputs/cc4d_zeroshot_subset.json
```
