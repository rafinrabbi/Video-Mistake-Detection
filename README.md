# Video-Mistake-Detection

Online (causal) procedural mistake detection in egocentric video, on
[CaptainCook4D](https://captaincook4d.github.io/captain-cook/).

A person follows a recipe on camera. The system processes the stream as
consecutive 4-second windows and decides, for each window and without access to
future frames, whether a mistake is happening. Each recipe has a task graph:
its steps plus precedence edges, forming a directed acyclic graph (DAG). The
code here studies how that graph should be used. It can be put into a
vision-language model's prompt, or it can be run as an explicit state estimator
over the person's progress through the recipe.

## Method

```
video window ─► content detector (Qwen2.5-VL-7B + LoRA) ─► P(mistake) ─────────┐
                                                                               ├─► fusion ─► mistake score
16-s causal look-back ─► step recogniser (LoRA) ─► P(step) ─► belief filter ───┘
                                                                ▲
                                                     recipe task graph (DAG)
```

- **Content detector** (`src/vmd/vlm/`). Qwen2.5-VL-7B with a LoRA adapter
  judges each window. Its score is the probability of the decision token, not
  the model's verbalised confidence. Prompt variants with and without the task
  graph live in `src/vmd/vlm/prompts/`.
- **Step recogniser** (`prompts/reason_stepreco.txt`). A second LoRA adapter
  sees a 16-s causal look-back (8 frames at 0.5 fps) and chooses the current
  step from a lettered menu. The softmax over the letter tokens is a
  distribution over the recipe's steps.
- **Partial-order belief filter** (`src/vmd/tracking/belief.py`). A forward
  filter keeps a belief over the current step and the set of completed steps.
  Its transition prior lets the person move only to steps whose prerequisites
  are done, with a small leak to forbidden steps and a transit state. The
  filter produces three structural signals:
  - `illegal_hold`: the person stays in a step whose prerequisites are unmet,
    or whose successors are already done;
  - `structural_surprise`;
  - `belief_entropy`.
- **Fusion** (`src/vmd/eval/fusion.py`). An L2-regularised logistic regression
  combines the content logit with the structural signals. It is fitted on the
  validation split only. Thresholds are chosen out of fold.
- **Topology controls** (`src/vmd/eval/matched_graph.py`). Linear-chain and
  graph-free filters use the same emissions and fusion as the DAG filter, so
  the topology is the only thing that differs between them.

## Repository layout

```
src/vmd/
  data/       CaptainCook4D loader, gold-graph export, causal window sampler
  graph/      ProceduralGraph schema (steps + precedence edges), builder, validator
  vlm/        Qwen2.5-VL wrapper, prompts, decision-token and step scoring
  tracking/   partial-order belief filter
  train/      LoRA fine-tuning (detector and step recogniser)
  eval/       causal protocol, event metrics (FROC), fusion, bootstrap curves
  decision/   smoothing and onset localisation
scripts/      experiment drivers (cc4d_*.py / .sh)
configs/      Hydra configs: paths, dataset, zero-shot and LoRA settings
data/graphs/  gold CaptainCook4D task graphs as ProceduralGraph JSON
data/splits/  unseen-recipe hold-out split
baseline/MistSense/  re-implementation of the MistSense baseline (ICCV 2025)
tests/        unit tests (no GPU or data needed)
```

## Setup

```bash
pip install -e .
pytest tests/ -v
```

Download the CaptainCook4D annotations, task graphs and 360p videos into
`data/raw/captaincook4d/`. Set the following environment variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face access for Qwen2.5-VL |
| `VMD_ROOT` | repository root, if it differs from the path in `configs/base.yaml` |
| `HF_HUB_DISABLE_XET=1` | only if large model downloads stall |

## Usage

```bash
# Fine-tune the content detector (GPU)
python scripts/cc4d_lora_train.py --output_dir outputs/lora_cc4d_8class

# Fine-tune the step recogniser (GPU)
python scripts/cc4d_lora_train.py --prompt_variant stepreco --epochs 1 \
    --neg_per_pos 0 --max_windows 8000 --context_sec 16 --context_fps 0.5 \
    --step_loss_weight 1.0 --gradient_checkpointing --output_dir outputs/stepreco

# Collect per-window predictions (GPU)
python scripts/cc4d_collect_preds.py --split test --stride 4 \
    --lora_adapter outputs/lora_cc4d_8class --save_preds outputs/preds_test.json

# Belief filtering, fusion and topology controls (CPU, from saved predictions)
python scripts/cc4d_matched_graph_experiment.py --help
python scripts/cc4d_threeway_fusion.py --help
python scripts/cc4d_structural_oracle.py --help
python scripts/cc4d_order_audit.py --help
```

All evaluation after prediction collection is CPU-only and replays saved
per-window predictions. The decision layer is selected on validation, frozen,
and applied to test once. Confidence intervals come from a bootstrap that
resamples recordings rather than windows.

## Evaluation protocol

- Strictly causal: a window's decision uses only frames up to its end.
- Official CaptainCook4D splits: 213 train, 62 validation and 109 test
  recordings.
- The eight native CaptainCook4D mistake classes are used
  (`src/vmd/data/mistake_classes.py`).
- Metrics: window-level AP and AUC, AUC per error type, event-level FROC, and
  operating-point metrics at validation-selected thresholds.
