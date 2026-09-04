# Experiments and tasks remaining before submission

Status board (visual): see the artifact link in the session notes, or read this file.
Last updated 2026-09-01. Hardware: one shared RTX PRO 6000, 96 GiB.

## Resource envelope (measured, 2026-09-01)

Both entry points now report peak VRAM and host RSS at exit, so these are
measurements rather than the estimates this table previously carried.

| Quantity | Measured | Note |
|---|---|---|
| Model weights, bf16 | **15.45 GiB** | 8.29 B params, no quantisation |
| **Inference peak VRAM** | **17.5 GiB** allocated / 18.2 reserved | was estimated 20–24; fits a 24 GiB card |
| **Inference peak host RSS** | **16.4 GiB** | **was estimated ~8 GiB — that was wrong by 2x** |
| **Training peak VRAM** | **55.9 GiB alloc / 67.5 GiB reserved** (torch, at exit) | gate raised 62→66→72 GB; reserved is what OOMs, not allocated |
| Training host RAM | ~12 GiB | 4 dataloader workers |
| CPU re-gates | 2–3 GiB, no GPU | replays a 9 MiB preds file |
| Disk | ~17 GiB | model cache 15.45 + adapter 0.35 |

Training cost is dominated by activations with **gradient checkpointing
disabled**, not by LoRA — its parameters and optimiser state together are under
0.6 GiB. Enabling checkpointing is the lever if training memory ever binds.

**Action:** raise `NEED_TRAIN_MIB` from 62000 to ~66000. G2a only started because
77 GiB happened to be free; at exactly 62 GiB free it would have been admitted
and then run out.

## Status at a glance

| | Item | Wall clock | VRAM · host RAM |
|---|---|---|---|
| ✅ | MistSense reproduced + retrained on CC4D | — | ~8 GiB · — |
| ✅ | R1/R2 zero-shot, graph prompt, full test | 14–24 h | ~22 GiB · 8 GiB |
| ✅ | R3/R4 LoRA graph-conditioned, full test (**main row**) | 14–24 h | ~22 GiB · 8 GiB |
| ✅ | LoRA fine-tune of the graph arm | ~3 h | 40–60 GiB · 12 GiB |
| ✅ | Decision-layer sweep, frozen on val | minutes | CPU · <2 GiB |
| ✅ | Statistical layer (bootstrap + McNemar) | minutes | CPU · ~3 GiB |
| ✅ | R9 protocol-matched baseline | minutes | CPU · ~3 GiB |
| ✅ | Confidence-source ablation (deviation-only) | seconds | CPU · <2 GiB |
| ✅ | Environment + resume validation | 4 min | ~22 GiB · 8 GiB |
| ✅ | **R5 graph-off inference** — DONE 2026-09-01 20:38 | 15.1 h | 17.5 GiB · 16.4 GiB |
| ✅ | **R6 no-graph adapter** — DONE 2026-09-02 22:38 | 7.5 h + 10.5 h | 61.7 / 17.5 GiB |
| ✅ | **R11 corrected pointer** — DONE 2026-09-04 04:57. F1 0.458 (≈ trivial), AUC 0.573, macro-F1 0.082; pointer aligns with gold on 7.7% of windows → OOD, fires on 93% | 3 h + 9 h | 67.5 / 18.2 GiB |
| ✅ | **Trivial always-fire baseline** — F1 0.462, EDS 1.000; every learned system within noise on F1 | CPU · sec | — |
| ✅ | **Matched-event onset** — R4 vs R6 median −4 s, CI [−8, 0]; earliness mostly firing rate | CPU · sec | — |
| ⏸ | R7 wrong-graph control — no longer bears on any claim | 7–24 h | 17.5 GiB |
| ✅ | Qualitative figure — DONE, Figure 2 built | 16 min | 17.5 GiB · 16.4 GiB |
| ⏸ | R10 static-dump arm — no longer bears on any claim | 7–24 h | 17.5 GiB |
| ⬜ | R8 baseline at native frame rate (optional) | minutes | CPU · ~3 GiB |
| ➖ | Offline protocol — **dropped** (saves 16–20 h) | — | — |
| ➖ | Multi-seed — **stated as limitation** (saves ~33 h) | — | — |
| ➖ | Second dataset — **scoped to cooking** | — | — |
| ➖ | Calibration as a contribution — **removed**; ECE now reported | — | — |
| ⬜ | Rationale-supervision retrain — future work, not this paper | ~3 h + 14–24 h | 40–60 GiB · 12 GiB |

**R5 result (2026-09-01):** removing the graph at inference costs +0.015 binary
F1, paired CI [−0.003, +0.035] — contains zero. AUC and macro-F1 unchanged. Only
earliness moves: EDS 0.806→0.697, CI [+0.022, +0.194], excluding zero.
Diagnosed cause: the step pointer saturates — 66% of all test windows are
conditioned on the *final* step, reached at a median 16% through a recording, so
the graph context is constant for two-thirds of the run
(`scripts/cc4d_pointer_analysis.py`).

**R6 result (2026-09-02):** a model trained *and* tested with no graph reaches
0.460 vs 0.482 — paired CI [−0.024, +0.065], contains zero. AUC and macro-F1
likewise unchanged. **Graph conditioning does not improve detection.** The
pre-registered pattern 1 obtained and the title was changed accordingly.

**What replicates:** earliness. EDS 0.806 → 0.697 (R5) → 0.673 (R6), paired CIs
[+0.022, +0.194] and [+0.054, +0.224], both excluding zero, at unchanged event
coverage (57/57/56 of 61). The graph tells the detector *when*, not *whether*.

**Second, larger defect found 2026-09-03 (CPU analysis, no run needed):**
training never used the advancing pointer at all. `lora_train.py` assigns
`step = graph.steps[min(window_index, n_steps-1)]` — the offline rule. With 9–27
steps and recordings of minutes, **92.8% of training windows carry the final step
as their graph context** (per-clip median 92.4%); positional variation exists only
in the first ~17 windows (~67 s). So:

1. the model was fine-tuned on a conditioning signal that was constant for 93% of
   its examples, and had almost no chance to learn to use step position;
2. training and evaluation used **different** step-assignment rules — a train/test
   mismatch in the conditioning itself.

This is a better explanation of the R5/R6 null than the eval-side saturation
alone, and it changes the cost of the fix.

**Highest-value next experiment — "the graph, delivered properly":**
| step | what | cost |
|---|---|---|
| 1 | one step-assignment rule, used in **both** train and eval; advance on step completion (or, cheapest credible version, proportional to elapsed time: `step = floor(t / (duration/n_steps))`) | code, hours |
| 2 | verify on CPU that the new rule leaves <20% of windows saturated, train and test | minutes |
| 3 | retrain the graph arm | ~3 h clear / ~7 h shared |
| 4 | one full test pass → new R4 | ~10–15 h |
| 5 | compare against the **existing** R5 and R6, which stay valid — neither uses the graph, so the pointer rule cannot affect them | CPU |

Total ~15–25 h GPU for a decisive answer, and no other row needs re-running.

**R11 result (2026-09-04):** the pointer fix made things worse instructively. Trained on always-correct gold steps, evaluated with a causal pointer right on only 7.7% of windows, the model fires on 93% regardless (0.958 aligned / 0.931 misaligned): out of distribution, not a mismatch detector. F1 0.458 ≈ trivial 0.462; macro-F1 0.185→0.082.

**Trivial baseline (2026-09-04):** fire-on-everything scores F1 0.462, EDS 1.000. R4 is +0.020 over it, CI [−0.050, +0.079] — not distinguishable. R6 and R11 are *below* it. Only AUC separates learned systems from trivial: +0.136, CI [+0.063, +0.198]. The earliness claim, controlled for detection (jointly detected events), shrinks to median −4 s, CI [−8, 0]. Paper retitled and reframed around ranking.

**Next experiment that would change the picture:** supervise step progress directly (a step-completion head or the model's own step_id, trained on gold spans), so the causal pointer can be delivered accurately. Nothing else currently queued.

**Status: all chains finished; nothing running; idle notifier watchers cleared
(2026-09-04).** `outputs/decision_v2/STOP_CHAIN` is in place. To resume:
`rm outputs/decision_v2/STOP_CHAIN` and relaunch the chain; it skips completed
stages, so it would start at R7 — which is cancelled, so there is nothing left
for it to do.

**Manuscript: complete. Zero placeholders, zero overfull boxes, 17 pages.**
Every registered test has a verdict in Table 3. Two values are left to the
authors and are macros at the top of `main.tex`: `\coderepo` (anonymised
artefact URL, at submission) and `\fundingnote` (at camera-ready). See
`paper/PLACEHOLDERS.md` for the closure record.

**Added 2026-09-04, no GPU needed:**
- **Strictly causal decision layer** (`scripts/cc4d_causal_smoothing.py`). The
  reported median filter is *centred*, so at k=5 it reads 8 s past each window —
  the one non-causal step in a paper claiming an online causal protocol. Priced
  by replaying the same saved predictions through a trailing filter of equal
  width: R4 F1 0.482 → 0.482 (+0.001), EDS 0.806 → 0.783; R5 0.467 → 0.464;
  R6 0.460 → 0.454. Ordering and every interval-based conclusion survive.
  Reported in §4.6, Limitations, and Appendix E. `decision_layer.py` untouched,
  so no saved row is invalidated.
- **Trivial baseline promoted to a real row** (`row_alwaysfire_test.json`),
  re-gated through the identical decision layer, so Table 1 and the register
  read it from an artefact instead of a hand-quoted number.
- **Complete experiment register** (Appendix D): R1–R12, including the two
  default-gate guard rows and the two cancelled controls.
- **Algorithm 1** — the online loop stated formally, with both `Advance` rules.
- **Table 3, the verdict table** — every registered reading against what the
  evidence licensed.

---

## 1. [CLOSED] Graph-conditioning ablation — rows R5 / R6 / R7

**Resolved 2026-09-02 (R5, R6) and 2026-09-04 (R11).** R7 cancelled.
The section below is kept as the record of why the experiment was run
and what reading was registered for it before it landed.

**Why.** This is the single load-bearing experiment for the paper's only claim.
The current headline compares *zero-shot + graph* against *fine-tuned + graph*, so
fine-tuning and graph conditioning move together. Without R5/R6 the result
supports "fine-tuning helps", not "the graph helps", and the title has to change.
Section~\ref{sec:graphablation} of the manuscript is a red placeholder until this
lands.

**Status.** `scripts/cc4d_graph_ablation_chain.sh` exists and is correct. It was
launched on 2026-07-28 and stalled inside the Hugging Face model download
(`logs/cc4d_graph_ablation_chain.log`, last line is a `Fetching 5 files` progress
bar). Nothing downstream ran. The `nograph` prompt template
(`src/vmd/vlm/prompts/reason_nograph.txt`) is written and correct.

**To run.**
```bash
# set HF_TOKEN first — the stall was an unauthenticated-download rate limit
export HF_TOKEN=...
nohup bash scripts/cc4d_graph_ablation_chain.sh &
```
The chain waits for a free GPU, smoke-tests on one clip, then runs G1 → G2a → G2b,
re-gating with R4's frozen decision params and rebuilding the table after each pass.

| Stage | What | Cost |
|---|---|---|
| S1 | smoke, nograph prompt, 1 clip | ~5 min |
| G1 → **R5** | graph-OFF inference, existing adapter, full test | 8–10 h GPU |
| G2a | retrain LoRA with nograph prompts, identical hyperparams | ~3 h GPU |
| G2b → **R6** | no-graph adapter, full test | 8–10 h GPU |

| G3 → **R7** | wrong-graph control, deranged graphs, full test | 8–10 h GPU |

**R7 needs no new code.** `scripts/cc4d_collect_preds.py` already implements
`--graph_permute_seed`, which deterministically maps every task to a *different*
task's graph. Verified on the test split: 0 fixpoints, all 24 tasks receive a
different graph, coverage unchanged at 109/109 recordings. G3 is now wired into
the chain and runs unattended after G2b.

**Cut lines.** R5 alone (CUT LINE A) is enough for a defensible ablation section.
R5+R6 (CUT LINE B) is the complete story. R7 (CUT LINE C) is the strongest
version — it is the row that answers "is it the plan's *content*, or just having
*some* structure in the prompt?" Total chain: ~30–35 h GPU.

---

## 1b. [FINDING, 2026-09-01] The fine-tuned model's free text is degenerate

Found by the 5-minute smoke test, not by a full run. On 76 consecutive test
windows every free-text field took a **single** value:

```
rationale             76/76  "Ground truth label."
observed_action       76/76  ""
expected_vs_observed  76/76  ""
observed_objects      76/76  []
uncertainty_reason    76/76  "none"
verbalized_confidence 76/76  0.9
confidence (token)    0.020 - 0.945      <- the only informative signal
```

**Cause, confirmed at source** (`src/vmd/train/lora_train.py:75-84`): the
supervision targets are built from labels alone and hardcode every free-text
field. Only `deviation` and `mistake_class` carry information. The model
reproduces the constants exactly -- the same failure mode as `confidence=0.9`,
extended to every text field.

**Consequence:** the manuscript's explainability claims were false and have been
removed (abstract, introduction, §6.3, conclusion), replaced by an explicit
disclosure in §4.3 and a Limitations entry. The qualitative figure must be built
from the score stream and typed decisions -- **not** from quoted rationales.

**Future work, not this paper:** CaptainCook4D ships free-text error
descriptions that could serve as genuine rationale targets. Supervising them
would make the explanation claim true and is a stronger paper, but it is a
retrain plus a full re-eval on top of the ablation chain. Do not queue it ahead
of the graph ablation.

---

## 2. [STRONG] Qualitative figure

**Why.** The paper claims the model "explains itself in language" and never shows a
single rationale. This is the cheapest reviewer-facing win available.

**Unblocked.** The saved prediction records carried only `confidence`,
`deviation`, `mistake_class` and the window bounds — the text fields were
computed and then discarded. `--log_text` now persists `rationale`,
`observed_action`, `observed_objects`, `step_id`, `expected_vs_observed`,
`uncertainty_reason` and `verbalized_confidence`. It is off by default so full
splits stay small. It also makes the verbalized-vs-token confidence comparison
recoverable, which it previously was not.

**To run** (as soon as a GPU is available), 2–3 recordings only:
```bash
python scripts/cc4d_collect_preds.py --split test --limit 3 --stride 4 \
    --lora_adapter outputs/lora_cc4d_8class --log_text \
    --save_preds outputs/decision_v2/qualitative_preds.json
```
~20–30 min GPU. Plot the score stream against gold error spans and quote three
rationales, **one of which must be a failure case**.

---

## 3. [STRONG] Read the related work properly

No compute. Currently `paper/main.tex` §2 carries three red editorial notes.

**DONE.** §2 is written and every citation verified.
- **Seminara et al.** — resolved: they use CaptainCook4D for *task-graph
  generation only*; their mistake detection is on Assembly101-O / EPIC-Tent-O and
  runs on action sequences, not video. No head-to-head on our benchmark exists
  to be made, and §2 now says so explicitly.
- **Spurio et al. (BMVC 2026)** — framed as contemporaneous work.
- **MoniTor** — record found (arXiv 2510.21449, NeurIPS 2025), but it is
  *surveillance* video anomaly detection (UCF-Crime, XD-Violence), not procedural
  mistake detection. Now cited as the anomaly-vs-procedural contrast. Its author
  list still needs expanding in refs.bib.
- **ASK-HINT** — no bibliographic record exists under that or any near name. The
  claim it supported was removed rather than cited speculatively.
- Assembly101, HoloAssist, EPIC-Tent, Ego4D bib entries added.

---

## 4. [NICE] Offline protocol — decide: re-run or drop

The old offline numbers (`outputs/full_eval_8class/ablation.md`) were produced
under the superseded decision layer and are not comparable to anything in the
current table. Offline step tracking also saturates at the final step for most
recordings, so the protocol as implemented is weak by construction.

**DONE — dropped (2026-09-01).** The paper is online-only. §5.3 now states the
reason in the text rather than omitting the protocol silently: offline tied the
expected step index to the window index, so it saturated at the final step for
most recordings and measured weaker graph conditioning rather than the benefit of
non-causal access. Saves 16–20 h GPU.

---

## 5. [NICE] Training-seed variance

Every interval in the paper is an *evaluation* bootstrap: it covers which
recordings were sampled, not which adapter was trained. The adapter was trained
once, on one seed.

**DONE — stated, not run (2026-09-01).** §Limitations now says plainly that the
adapter was trained once with one seed, that the intervals cover evaluation
sampling and not training variance, and that the per-class numbers in particular
should be read as one run. Saves ≈33 h GPU. Justification: the zero-shot→LoRA gap
is 0.332 with a [0.246, 0.409] interval, far wider than plausible seed variance.

---

## 6. [NICE] Second dataset

The claim is currently cooking-only, on one benchmark. A transfer study needs a
benchmark that ships gold task graphs, which is the binding constraint —
EgoPER and Assembly101-O do not. Realistically out of scope for this cycle;
scope the abstract to cooking instead.

---

## 7. Housekeeping

- [ ] `git add` the untracked decision_v2 pipeline (`scripts/cc4d_calibrate.py`,
      `cc4d_decision_*.py`, `cc4d_bootstrap_compare.py`, `cc4d_mistsense_retile.py`,
      `src/vmd/eval/decision_layer.py`, `tests/test_decision_layer.py`) — the paper's
      reproducibility statement points at code that is not committed.
- [ ] Stale docs still describe EgoOops / 6 classes / a running eval:
      `CLAUDE.md`, `RESULTS_SUMMARY.md`, `pipeline_diagram.md`, `docs/*.txt`,
      `PROGRESS_REPORT.md` §9–10.
- [ ] `pytest tests/ -v` — confirm the suite still passes after the decision_v2 work.
- [ ] Anonymised code release URL for the reproducibility statement.
- [ ] Swap `paper/main.tex` to the official venue style once the venue is fixed.

---

## Completed this session

- **R9, protocol-matched baseline** — MistSense per-frame probabilities re-tiled
  onto the identical 23,696 windows (all 109 recordings, zero missing dumps),
  decision layer tuned on validation ($\tau=0.7$) and applied to test once:
  **0.477** binary F1 vs our 0.482, paired bootstrap $+0.005$ [$-0.060$, $+0.059$].
  This replaces the oracle-threshold 0.495 that previously appeared to beat us.
  Artefacts: `outputs/decision_v2/row_mistsense_tuned_test.json`,
  `bootstrap_mistsense_tuned.json`, `bootstrap_lora_vs_mistsense.json`.
- **Deviation-only decision ablation** — recomputed on CPU from saved predictions:
  `outputs/decision_v2/row_lora_devonly_test.json`.

---

## Interruption and resume (shared GPU)

Every long stage is now restart-safe, at these granularities:

| Stage | Granularity | Lost on interrupt |
|---|---|---|
| Inference (`cc4d_collect_preds.py --resume`) | one recording | ≤1 clip, ~4–13 min |
| LoRA training (`cc4d_lora_train.py --resume`) | 50 optimiser steps (`--checkpoint_every`) | ≤50 steps |

**After a reboot or a kill, re-run the same command.** The chain passes
`--resume` to every stage, so restarting it is safe and cheap: finished stages
re-verify in about a minute and finished clips are skipped without re-inference.

```bash
setsid nohup env HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 \
    bash scripts/cc4d_graph_ablation_chain.sh > /dev/null 2>&1 < /dev/null &
```

How it works, and why it is safe:

- **Inference.** A clip's per-window records are appended to the
  `*_preds.jsonl` sidecar in a single write *after* the clip completes, so every
  clip in that file is whole. Per-clip state (the graph step pointer and the
  three-window judgement history) is initialised inside the clip loop and never
  crosses clips, so skipping a finished clip reproduces the uninterrupted run
  exactly. `tests/test_resume.py` pins that equivalence, including a torn final
  line and the gold-event bookkeeping for skipped clips.
- **Training.** Epoch order comes from a seeded permutation rather than
  `DataLoader(shuffle=True)`, so a resumed epoch replays the same order and skips
  the completed prefix *by index*, without decoding frames it would discard.
  Checkpoints are written only on gradient-accumulation boundaries (optimiser
  state, scheduler state, adapter weights), so a resume is a clean restart rather
  than a dropped partial batch, and they are written to a temp file and renamed
  so an interrupted write cannot replace a good checkpoint with a truncated one.
  The state file is deleted on successful completion, so `--resume` on a finished
  run does not restart it mid-way.

**Caveat: the chain itself does not survive a reboot** — it is an ordinary
background process. Re-run the command above after each reboot, or install a
`@reboot` crontab entry to have it restart itself.

**Do not edit `scripts/cc4d_graph_ablation_chain.sh` while it is running.** Bash
reads a script incrementally by byte offset, so editing in place can make the
running instance execute garbage. Stop it, edit, relaunch — resume makes that
nearly free.
