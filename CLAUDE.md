# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project

Online procedural mistake detection in egocentric video. A vision--language
model is conditioned on the task's procedural graph and judges each causal
window: is this a deviation, of what type, and when did it start.

**Benchmark: CaptainCook4D** (NeurIPS 2024 D&B) — 384 recordings, 24 recipes,
gold task graphs, temporally bounded typed errors. EgoOops was the original
benchmark and was **removed** in `208bb5c`; the baseline collapsed on it at that
scale. `data/egoops.py` survives only as the home of the `EgoOopsClip` /
`MistakeAnnotation` dataclasses — the names are legacy, the data is not used.

**Target venue:** ACL ARR → EMNLP 2026. See `VENUE_RECOMMENDATION.md`.

**Scope of the paper (decided Aug 2026):** a *single* contribution — procedural
graph-conditioned VLM reasoning. The uncertainty-adaptive threshold that was
once "Contribution B" was falsified at full scale and is **out of the
manuscript**; its sources were deleted and are recoverable at `141385c`. Do not
reintroduce it, or calibration, into the paper without being asked.

## Commands

```bash
pip install -e .
pytest tests/ -v                     # 49 tests, no GPU or data needed

# Collect per-window predictions (GPU). --lora_adapter selects the fine-tuned
# arm; --prompt_variant nograph and --graph_permute_seed drive the ablations.
python scripts/cc4d_collect_preds.py --split test --stride 4 \
    --lora_adapter outputs/lora_cc4d_8class \
    --save_preds outputs/decision_v2/lora_test_preds.json

# LoRA fine-tune (GPU, ~3 h)
python scripts/cc4d_lora_train.py --output_dir outputs/lora_cc4d_8class

# Everything below is CPU-only, replayed from saved predictions:
python scripts/cc4d_decision_sweep.py   --val_preds ... --test_preds ... --arm lora
python scripts/cc4d_decision_eval.py    --preds ... --split test --tau ... --out ...
python scripts/cc4d_bootstrap_compare.py --preds ... --params ... --label ...
python scripts/cc4d_build_ablation.py   --manifest outputs/decision_v2/ablation_manifest.json

# Graph ablation chain (GPU, ~30-35 h): waits for a free GPU, then G1/G2/G3
bash scripts/cc4d_graph_ablation_chain.sh    # cancel: touch outputs/decision_v2/STOP_CHAIN

# Paper
python paper/gen_tables.py && (cd paper && latexmk -pdf main.tex)
```

CLI (`vmd`): `build-graph`, `infer`, `eval`, `export-cc4d-graphs`, `train`.

## Architecture

```
CaptainCook4D recording
  → data/captaincook4d.py        loader → clips + typed annotations
  → data/captaincook4d_graphs.py gold DAG → ProceduralGraph (cached)
  → data/sampler.py              4 s windows, stride 4 s, 1 fps, causal
  → vlm/qwen25vl.py              graph-conditioned prompt → structured JSON
                                 + decision-token confidence (overrides the
                                   verbalized value, which is degenerate)
  → eval/decision_layer.py       gate → smooth → metrics  [replayable on CPU]
  → decision/postprocess.py      median smoothing + onset localization
  → eval/metrics.py              F1, macro-F1, AUC, EDS, start-MAE, ECE, Brier
```

Online step tracking is the substance of the method: the expected-step pointer
advances by one whenever the model reports *no* deviation, so the graph is
walked as a state machine driven by the model's own judgements.

`eval/decision_layer.py` is the single source of truth for how predictions
become decisions. Its contract must reproduce a GPU run to float precision —
if you change it, you invalidate every saved row.

## Key types

- `ProceduralGraph` (`graph/schema.py`): pydantic; `Step`/`Object`/`Action`/`State`/`Edge`.
  **Gold CC4D graphs populate only steps and precedence edges** — objects,
  states and variations are empty on every recipe. Do not describe the
  conditioning as richer than that.
- `VLMPrediction` (`vlm/qwen25vl.py`): parsed JSON; `confidence` is the
  decision-token probability, `verbalized_confidence` the model's own claim.
- `CaptainCook4DDataset` (`data/captaincook4d.py`), `MISTAKE_CLASSES`
  (`data/mistake_classes.py` — the taxonomy's single source of truth).

## Mistake classes (CC4D native, 8)

`no_mistake | order_error | technique_error | preparation_error |
measurement_error | missing_step | timing_error | temperature_error`

`missing_step` is **structurally unreachable** by a windowed detector: those
annotations are exactly the non-localizable ones, so they mark no window. Test
support is 2 windows and F1 is 0.000 in every row. Report macro-F1 both with
and without it.

## Data layout

```
data/raw/captaincook4d/    annotations + videos (360p) + task_graphs
data/graphs/<recipe>.json  gold DAG → ProceduralGraph (24 used, 29 cached)
outputs/lora_cc4d_8class/  current adapter
outputs/decision_v2/       current results: row_*, bootstrap_*, ablation.{json,md}
outputs/full_eval_8class/  saved per-window predictions (the re-gate inputs)
paper/                     manuscript; tables generated from the JSONs above
```

## Configs

`base.yaml` (paths, model ids, wandb) · `cc4d.yaml` (dataset) ·
`qwen25vl_zeroshot.yaml` · `qwen25vl_lora.yaml`

## Evaluation protocol

- **Online causal** is the reported protocol: no future access.
- Offline exists in history but its step conditioning saturates; current offline
  numbers predate the frozen decision layer and are **not reported**.
- The decision layer is selected on validation, frozen, and applied to test
  **once**. Each arm gets its own tuned layer.
- Every headline number carries a clip-level bootstrap CI (n=1000, seed 0);
  recordings, not windows, are the resampling unit.

## Env vars

- `HF_TOKEN`: required. This host also needs `HF_HUB_DISABLE_XET=1` — the Xet
  backend stalls at zero bytes on multi-GB files here (see
  `scripts/fetch_qwen_shards.sh`).
- `OPENAI_API_KEY`: LLM graph building only; all reported results use the gold
  CC4D graphs.
- `VMD_ROOT`, `VMD_GRAPH_MODEL`, `WANDB_API_KEY`.

## Status and open work

`EXPERIMENTS_TODO.md` is authoritative. **No blockers; nothing is running.**

The graph ablation landed and came back negative: R5 (graph off at inference)
and R6 (graph off in training and inference) are both within noise of the main
arm, and R11 (the corrected step pointer, retrained) is worse. The title was
changed to match. R7 and R10 were pre-registered and cancelled unrun, because
both decompose a detection effect R5/R6 showed does not exist.

The manuscript is complete: zero placeholders, zero overfull boxes, 17 pages,
every registered test given a verdict in its Table 3. Two macros at the top of
`paper/main.tex` are left for the authors — `\coderepo` and `\fundingnote`.

The next experiment that would change the picture is supervising step progress
directly, so the causal pointer can be delivered accurately. It is not queued.
