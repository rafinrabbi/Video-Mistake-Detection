# MistSense — Reproduction Notes

Paper: Patsch et al., "MistSense: Versatile Online Detection of Procedural and Execution Mistakes," ICCV 2025.
Source: CVF Open Access (no arxiv version found at the time of writing).
Official code: **Not released publicly.** No GitHub link or project page in the paper.

This document records every implementation choice we had to make because the paper either omitted, partially specified, or used a "following [X]" reference. Anyone wishing to validate or reproduce the architecture should review this file first.

---

## Architecture choices

| Item | Paper status | Our choice | Justification |
|---|---|---|---|
| Visual encoder identity | "ViT-L" (§4.6) — pretraining not stated | EVA-ViT-L (BLIP-2 family) | BLIP-2 ref [20] uses EVA-ViT, ViT-L variant matches paper's "L" size. CLIP-ViT-L/14 is an alternative. |
| Base Q-Former weights | "similar to the one proposed in Blip2" (§3.1) | `Salesforce/blip2-opt-2.7b` Q-Former, frozen | Closest open BLIP-2 release. |
| `t_q` (num learnable queries) | UNSPECIFIED | 32 | BLIP-2 default. |
| `d_2` (hidden dim) | UNSPECIFIED | 768 | BLIP-2 / BERT-base default. |
| `d_1` (visual feat dim) | UNSPECIFIED | 1024 | ViT-L patch embed dim. |
| `t_s` (segment length) | UNSPECIFIED | 8 frames | 4 s at 2 fps (§4.6 fps stated). |
| Video Q-Former depth | "BERT encoder" (§3.1) | 12 layers, 12 heads | BERT-base default. |
| Hand Q-Former | "same structure as the Video Q-Former" | identical | direct quote. |
| Hand pose encoder | "adapt from [32] ... TCNs [17] ... self-attention" | Per-joint Conv1d×3, then joint-wise self-attention, mean over joints, linear → `d_h` | follows [Shamil 2024, ECCV'24]. |
| `d_h` (hand feat dim) | UNSPECIFIED | 256 | reasonable for 21 joints. |
| TCN width/depth | UNSPECIFIED | 128 ch × 3 layers | match Lea 2017 family default. |
| `v` (joints), `m` (hands), `c` (coord) | symbol-only | 21, 2, 3 | Mediapipe Hands defaults [23]. |
| Positional embedding (video) | not stated | learnable 1D, length `t_s` | analogous to hand branch which is stated. |
| Positional embedding (hand) | "framewise positional embedding is added" (§3.1) | learnable 1D | quote-supported. |
| Mistake head | "the mistake classification layer" (§3.1) | Linear(`d_2`, 1) per query token | minimum that yields `m ∈ R^{t_q}`. |
| Fusion operation | "added together" (§3.1) | element-wise sum | direct quote. |
| Gating `g_m` formulation | "where σ is the sigmoid function" (§3.2); explicit equation omitted as image | `g_m = max_q σ(logit_q)` | per-query → per-segment aggregator. Mean is an alternative. |
| Threshold `τ` | numeric value not stated (§3.2) | 0.5 | standard binary decision threshold; tune on val. |
| LLM projection layer | "linear layer that maps to LLM embedding space" (§3.2) | Linear(`d_2`, 4096) | quote-supported (4096 = Llama2-7B hidden). |
| Activation (Q-Former FFN) | UNSPECIFIED | GELU | BERT/BLIP-2 default. |
| Norm placement (Q-Former) | UNSPECIFIED | Post-norm | BERT default. |
| LoRA rank / alpha | UNSPECIFIED (§3.3 mentions LoRA [12] only) | r=16, α=32, dropout=0.05, target=q_proj,v_proj | common Llama2 finetune defaults. |

---

## Training choices

| Item | Paper status | Our choice | Notes |
|---|---|---|---|
| Optimizer | "AdamW" (§4.6) | `torch.optim.AdamW` | — |
| Adam betas / eps | UNSPECIFIED | (0.9, 0.999), 1e-8 | PyTorch defaults |
| LR (main) | "3e-5" (§4.6) | 3.0e-5 | — |
| LR (hand encoder) | "2.5e-2 without weight decay" (§4.6) | param-group lr=2.5e-2, wd=0 | — |
| Weight decay (main) | "5e-3" (§4.6) | 5.0e-3 | — |
| LR schedule | "cosine ... linear warmup starting from 1e-6" (§4.6) | cosine + linear warmup | warmup duration not given → 5% of total steps |
| Epochs | 12 (§4.6) | 12 | — |
| Batch size | 16 (Epic-tent) / 32 (HoloAssist) (§4.6) | per-dataset config files | — |
| Mixed precision | UNSPECIFIED | bf16 | A6000 supports bf16; matches "2 RTX A6000" mention |
| Grad clip norm | UNSPECIFIED | 1.0 | safe default |
| Random seed | UNSPECIFIED | not pinned in scaffold | set in your launch script |
| Class weights | Epic-tent w_0=w_1=1; HoloAssist w_0=1, w_1=4 (§4.6) | per-config | — |

### Three-stage schedule (§3.3)
Implemented as `apply_stage(model, stage=k)` for k∈{1,2,3}.
Stage 2 requires the explanation-text decoder pipeline (LLM frozen, projection trainable, CE loss). Stage 3 adds `peft.LoraConfig` over the LLM. Both stages depend on an external LLM loader and tokenized explanation dataset — provided as TODO points in `train.py` because Llama2-7B needs HF auth and is not auto-fetched.

---

## Data choices

| Item | Paper status | Our choice | Notes |
|---|---|---|---|
| Datasets | Epic-tent-O (open-set) and HoloAssist (§4.1) | Skeleton loader for both; user provides per-clip features | gated downloads, not auto-fetched |
| Frame rate | 2 fps (§4.6) | precompute features at 2 fps | extraction pipeline out of scope |
| Image preprocessing | UNSPECIFIED | 224×224, ImageNet norm | BLIP-2 default |
| Train extension on Epic-tent | full unmodified Epic-tent added to training set (§4.2) | document only — user manages splits | — |
| HoloAssist explanations | from dataset (§4.3) | load `explanation.txt` per clip | — |
| Epic-tent explanation generation | "Place guyline was done in the wrong order" template (§4.3) | template string in TODO comment | not implemented in skeleton |

---

## What is implemented

- Stage 1 — weighted BCE detection (Q-Formers + hand encoder + mistake head)
- Stage 2 — explanation CE on `llm_proj` (frozen Q-Formers + frozen LLM)
- Stage 3 — LoRA finetune of LLM via `peft.get_peft_model` (frozen `llm_proj`)
- Mediapipe hand-keypoint extraction (Tasks API) — `scripts/extract_features.py`
- CLIP-ViT-L/14 visual feature extraction — `scripts/extract_features.py` ([DEVIATION] from "BLIP-2 ViT-L"; see below)
- LLM tokenization + soft-prefix CE forward — `src/llm.py::build_llm_inputs`
- LLM generation w/ visual prefix — `src/llm.py::generate_explanation`
- Caption metrics (BLEU / ROUGE-L / CIDEr) — `src/caption_metrics.py` via `sacrebleu`, `rouge_score`, `pycocoevalcap`
- Online causal sliding-window evaluator — `scripts/eval_online.py`
- Per-clip eval + τ sweep — `scripts/eval_test.py`
- Caption eval w/ optional LoRA — `scripts/eval_caption.py`

## What is NOT implemented

- Llama2-7B specifically — substituted with OPT-2.7b from cached `Salesforce/blip2-opt-2.7b` ([DEVIATION] below)
- Codabench HoloAssist submission packaging
- Other paper baselines (TSformer, PREGO, DO, MSG², GazeCompl, Count-Based)
- Epic-tent-O and HoloAssist datasets (gated; this repo uses EgoOops)
- Per-query fine-grained mistake supervision (we use segment-binary labels broadcast over `t_q`)

---

## How to validate against the paper

1. **Architecture sanity:** instantiate `MistSense(ModelConfig())`, forward a dummy `(B=2, t_s=8, d_1=1024)` visual tensor and `(B=2, c=3, t_s=8, v=21, m=2)` hand tensor; verify `logits.shape == (2, t_q)`. See `notebooks/walkthrough.ipynb`.
2. **Loss sanity:** `WeightedBCELoss(w0=1, w1=4)` should equal `BCEWithLogitsLoss(pos_weight=4)` up to averaging in expectation when w_0=1.
3. **Detection F1 on Epic-tent-O:** paper reports 59.8 RGB-only, 63.3 RGB+Hand (Table 3). Reproduce after feature extraction + 12 epochs.
4. **Detection F1 on HoloAssist:** paper reports 54.8 RGB-only, 55.7 RGB+Hand (Table 1) on the benchmark server.

---

## EgoOops adaptation results (this repo)

Paper used Epic-tent-O / HoloAssist (gated). This project's dataset is EgoOops (50 clips: 30 train / 10 val / 10 test). Mistake labels binarized to {0,1} from EgoOops `mistake_class` (`no_mistake` → 0, anything else → 1).

### Feature extraction
- Visual: **CLIP-ViT-L/14** CLS embedding (1024-d) instead of BLIP-2 (no public BLIP-2 with ViT-L; ViT-g variant was 10x slower). Documented as [DEVIATION] in `scripts/extract_features.py`.
- Hand: Mediapipe Tasks `HandLandmarker` (legacy `mp.solutions` API removed in mediapipe 0.10.35 / py 3.13). Model: `hand_landmarker.task` float16.
- Sampling: 2 fps (§4.6).
- Output: `data/processed/mistsense/<clip_id>/{visual.npy, hand.npy, labels.npy, explanation.txt}` (~209 MB total).

### Stage 1 results

Two runs:

**Run 1** (`stage1.pt`, w0=1, w1=4, final-ckpt save):
- Best val f1_mistake = 0.39 (epoch 6)
- Final E12 val f1_mistake = 0.18
- Test (final ckpt): f1_mistake = **0.00** across τ ∈ [0.3, 0.7] (model collapsed to all-correct)

**Run 2** (`stage1_best.pt`, w0=1, w1=8, best-val-ckpt save):
- Best val f1_mistake = 0.36 (epoch 2)
- Test (best ckpt): f1_mistake = **0.00**; f1_overall up to 0.48 at τ=0.7 (still predicting all-correct)

### Why so far from paper numbers (paper Epic-tent-O F1=63.3)

- Train set 30 clips vs paper's full Epic-tent + supplementation
- Segment-level binary label vs per-frame fine-grained (paper §3.3 supervises per-query)
- Dataset shift: EgoOops actions (ionic-reaction, calligraphy, etc.) differ from tent assembly
- Pretrained ViT-L is generic CLIP, not BLIP-2-finetuned visual encoder
- Train/val/test split has near-zero positive overlap; val/test mistake-segment counts very small
- Best ckpt picked by val f1_mistake; small val set means high variance → overfit to val noise
- Hand-pose modality contributes — but on EgoOops most clips lack persistent hand presence (different camera angle from paper datasets)
- Cosine LR cools too aggressively for 12-epoch / small-dataset regime

These are characteristic small-dataset reproduction caveats. The pipeline (extraction → architecture → loss → training schedule) executes end-to-end and matches paper structure.

### Stage 2 results (`stage2_best.pt`, llm_proj only)

Train 30 clips (only those with non-empty explanations), val on EgoOops val.
LLM: OPT-2.7b extracted from `Salesforce/blip2-opt-2.7b`. llm_proj shape = (768, 2560).

| Epoch | train_ce | val_ce |
|------:|---------:|-------:|
| 1     | 4.67     | 5.15   |
| 6     | 2.87     | 3.94   |
| 12    | 2.48     | **3.71** |

Best ckpt = epoch 12 (val_ce=3.71); saved at every improvement (monotonic).

### Stage 3 results (`stage3_best.pt` + `stage3_best_lora/`, LoRA)

LoRA r=16, α=32, dropout=0.05, target=(q_proj, v_proj). Trainable: 5.24M / 2.66B (0.20%).

| Epoch | train_ce | val_ce |
|------:|---------:|-------:|
| 1     | 2.41     | 3.47   |
| 5     | 1.15     | **3.01** |
| 12    | 0.67     | 3.06   |

Val plateaus at E5 then mild overfit. Best ckpt = E5 (val_ce=3.01).

### Caption metrics (test, 4 clips with non-empty references)

| | BLEU | ROUGE-L | CIDEr |
|-|-----:|-------:|------:|
| Stage 2 (no LoRA) | 1.02 | 0.037 | ~0 |
| Stage 3 (LoRA)    | **1.16** | **0.123** | **0.068** |

LoRA improves all three. Generated text remains generic ("grasp a pair of …"); 4 test references is too few for reliable BLEU/CIDEr. Paper Epic-tent-O reports BLEU ≈ 7.6, ROUGE-L ≈ 30.5, CIDEr ≈ 51.6 (Table 4) — gap reflects 7× less training data, OPT-2.7b vs Llama2-7B, and EgoOops's much shorter / template-free references.

### Online sliding-window detection (test, all frames)

10 clips, 8880 frames, 1074 positives. Causal: at frame t, model sees [t-t_s+1 .. t] only (zero-padded if t < t_s-1).

| τ    | F1_overall | F1_mistake | P_mistake | R_mistake | F1_correct |
|-----:|-----------:|-----------:|----------:|----------:|-----------:|
| 0.30 | 0.394 | 0.031 | 0.022 | 0.051 | 0.756 |
| 0.40 | 0.406 | 0.027 | 0.021 | 0.041 | 0.785 |
| 0.50 | 0.428 | 0.028 | 0.023 | 0.034 | 0.829 |
| 0.60 | 0.453 | 0.018 | 0.022 | 0.015 | 0.889 |
| 0.70 | 0.466 | 0.000 | 0.000 | 0.000 | 0.932 |

F1_mistake non-zero across most thresholds (per-frame eval is finer-grained than per-window `eval_test.py`). Still far below paper Epic-tent-O F1=63.3 for the same reasons enumerated under §"Why so far from paper numbers".

### Reproduction commands

```bash
PROJ=<project>

# 0. Extract features (~50 min on RTX Pro 6000)
python -u scripts/extract_features.py \
  --videos_dir "$PROJ/data/raw/egoops/videos" \
  --annot_dir  "$PROJ/data/raw/egoops/annotations" \
  --out_dir    "$PROJ/data/processed/mistsense"

# 1. Stage 1 — detection (~3 min)
python -u -m src.train --stage 1 \
  --base configs/base.yaml --override configs/egoops.yaml \
  --data_root   "$PROJ/data/processed/mistsense" \
  --train_annot "$PROJ/data/raw/egoops/annotations/train.json" \
  --val_annot   "$PROJ/data/raw/egoops/annotations/val.json" \
  --save stage1_best.pt

# 2. Stage 2 — train llm_proj (freeze Stage-1 backbone + LLM)
python -u -m src.train --stage 2 \
  --base configs/base.yaml --override configs/egoops.yaml \
  --data_root   "$PROJ/data/processed/mistsense" \
  --train_annot "$PROJ/data/raw/egoops/annotations/train.json" \
  --val_annot   "$PROJ/data/raw/egoops/annotations/val.json" \
  --stage1_ckpt stage1_best.pt --save stage2_best.pt

# 3. Stage 3 — LoRA finetune LLM (freeze llm_proj)
python -u -m src.train --stage 3 \
  --base configs/base.yaml --override configs/egoops.yaml \
  --data_root   "$PROJ/data/processed/mistsense" \
  --train_annot "$PROJ/data/raw/egoops/annotations/train.json" \
  --val_annot   "$PROJ/data/raw/egoops/annotations/val.json" \
  --stage1_ckpt stage1_best.pt --stage2_ckpt stage2_best.pt \
  --save stage3_best.pt
# LoRA adapter saved to ./stage3_best_lora/

# 4. Eval — per-window detection
python -u scripts/eval_test.py \
  --data_root   "$PROJ/data/processed/mistsense" \
  --test_annot  "$PROJ/data/raw/egoops/annotations/test.json" \
  --ckpt stage1_best.pt

# 5. Eval — online causal per-frame
python -u scripts/eval_online.py \
  --data_root   "$PROJ/data/processed/mistsense" \
  --test_annot  "$PROJ/data/raw/egoops/annotations/test.json" \
  --ckpt stage1_best.pt --out outputs/online_eval.json

# 6. Eval — explanation captioning (Stage 2 or Stage 3+LoRA)
python -u scripts/eval_caption.py \
  --data_root   "$PROJ/data/processed/mistsense" \
  --test_annot  "$PROJ/data/raw/egoops/annotations/test.json" \
  --stage1_ckpt stage1_best.pt --stage2_ckpt stage3_best.pt \
  --lora_dir stage3_best_lora --out outputs/caption_stage3.json
```

### LLM substitution rationale (`llm_pretrained`, `llm_hidden`)

`configs/egoops.yaml` sets:
```yaml
model:
  llm_pretrained: "Salesforce/blip2-opt-2.7b"
  llm_from_blip2: true
  llm_hidden: 2560
```
This routes `src/llm.py::build_llm` to extract `OPTForCausalLM` (d=2560) out of the BLIP-2 wrapper instead of fetching gated Llama2-7B (d=4096). To target Llama2-7B set `llm_pretrained: meta-llama/Llama-2-7b-hf`, `llm_from_blip2: false`, `llm_hidden: 4096`, and supply HF auth.

## Open questions worth contacting authors about

- Exact `t_q`, `d_2`, `t_s` values.
- Exact gating equation `g_m` formula (only "σ" mentioned; aggregation over `t_q` queries left open).
- Exact `τ` value used in §3.2.
- Hand-pose encoder TCN exact depth/width.
- LoRA rank / alpha / target modules.
- Warmup duration.
