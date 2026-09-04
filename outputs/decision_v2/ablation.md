# CC4D 8-class ablation — online causal, all test clips

Test clips: 109 | nonlocalizable: 69 | untyped dropped: 1

| row | binary_f1 | macro_f1 | macro_f1_excl_missing_step | auc_raw | precision | recall | eds | start_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 zero-shot, graph prompt, default gate (tau .5, and, k3) | 0.036 [0.019,0.056] | 0.107 [0.100,0.114] | 0.122 | 0.480 [0.447,0.512] | 0.259 | 0.019 | 0.245 [0.150,0.356] | 236.197 |
| R2 zero-shot, graph prompt, val-tuned gate | 0.150 [0.127,0.172] | 0.108 [0.100,0.115] | 0.123 | 0.480 [0.447,0.512] | 0.264 | 0.105 | 0.985 [0.971,0.996] | 64.399 |
| R3 LoRA, graph prompt, default gate (tau .5, and, k3) [guard row] | 0.471 [0.384,0.548] | 0.185 [0.161,0.208] | 0.211 | 0.636 [0.563,0.698] | 0.408 | 0.556 | 0.763 [0.656,0.861] | 60.162 |
| R4 LoRA, graph prompt, val-tuned gate (MAIN) | 0.482 [0.392,0.558] | 0.185 [0.161,0.208] | 0.212 | 0.636 [0.563,0.698] | 0.407 | 0.589 | 0.806 [0.705,0.892] | 60.443 |
| R5 LoRA, graph-OFF prompt (inference ablation), R4 gate | 0.467 [0.383,0.536] | 0.191 [0.166,0.211] | 0.218 | 0.631 [0.564,0.688] | 0.419 | 0.527 | 0.697 [0.591,0.801] | 81.544 |
| R6 no-graph LoRA (trained + evaluated graph-OFF), R4 gate | 0.460 [0.381,0.534] | 0.186 [0.165,0.206] | 0.213 | 0.629 [0.560,0.692] | 0.453 | 0.467 | 0.673 [0.565,0.778] | 79.629 |
| R11 LoRA retrained, rate-limited step pointer (corrected conditioning), R4 gate | 0.458 [0.377,0.542] | 0.082 [0.061,0.104] | 0.094 | 0.573 [0.484,0.658] | 0.303 | 0.933 | 0.834 [0.741,0.912] | 67.258 |
| R9 MistSense re-tiled to 4s grid, val-tuned gate | 0.477 [0.396,0.557] | 0.103 [0.096,0.108] | 0.118 | 0.585 [0.539,0.630] | 0.325 | 0.893 | 0.991 [0.985,0.997] | 60.927 |
| R12 TRIVIAL fire-on-every-window baseline | 0.462 | 0.185 | 0.212 | 0.500 | 0.300 | 1.000 | 1.000 | 65.071 |

### Per-class F1 (appendix)

| row | f1_no_mistake | f1_order_error | f1_technique_error | f1_preparation_error | f1_measurement_error | f1_missing_step | f1_timing_error | f1_temperature_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 zero-shot, graph prompt, default gate (tau .5, and, k3) | 0.813 | 0.010 | 0.028 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| R2 zero-shot, graph prompt, val-tuned gate | 0.812 | 0.012 | 0.029 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| R3 LoRA, graph prompt, default gate (tau .5, and, k3) [guard row] | 0.705 | 0.326 | 0.167 | 0.058 | 0.074 | 0.000 | 0.151 | 0.000 |
| R4 LoRA, graph prompt, val-tuned gate (MAIN) | 0.696 | 0.327 | 0.167 | 0.060 | 0.082 | 0.000 | 0.150 | 0.000 |
| R5 LoRA, graph-OFF prompt (inference ablation), R4 gate | 0.725 | 0.284 | 0.204 | 0.075 | 0.124 | 0.000 | 0.116 | 0.000 |
| R6 no-graph LoRA (trained + evaluated graph-OFF), R4 gate | 0.751 | 0.277 | 0.161 | 0.066 | 0.142 | 0.000 | 0.095 | 0.000 |
| R11 LoRA retrained, rate-limited step pointer (corrected conditioning), R4 gate | 0.147 | 0.250 | 0.060 | 0.059 | 0.125 | 0.000 | 0.019 | 0.000 |
| R9 MistSense re-tiled to 4s grid, val-tuned gate | 0.823 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R12 TRIVIAL fire-on-every-window baseline | 0.696 | 0.327 | 0.167 | 0.060 | 0.082 | 0.000 | 0.150 | 0.000 |

### Footnotes

1. Decision configs: 'default' = tau .5 / AND-gate / median k3 / persistence 2 (historical). 'val-tuned' = selected on the 62-clip val split by binary F1 (persistence by val EDS), frozen, applied once to test. R5-R7 reuse R4's frozen gate so the only variable is the prompt/graph conditioning.
2. MistSense's originally reported 0.495 used a tau sweep ON TEST (oracle selection) over 872 coarse windows; R8/R9 replace it with val-selected tau. The oracle number is retained here for transparency, not comparison.
3. Zero-shot rows: auc_raw ~= 0.48 (chance) — the tuned zero-shot F1 reflects threshold placement on a near-chance ranker, and its high EDS reflects indiscriminate early firing (see EDS footnote).
4. R7 and R10 were pre-registered and cancelled without being run: both decompose a detection effect, and R5/R6 established there is none to decompose. Their row files do not exist and the builder skips them.
5. R12 is the trivial always-fire detector, re-gated through the identical decision layer for comparability. Its auc_raw is 0.5 by construction (a constant score cannot rank).
6. R11 corrects the step-assignment rule on BOTH sides (gold-step teacher forcing in training, rate-limited causal pointer at eval). It cuts saturation from 66% to 20.7% of windows and still fails: the causal pointer agrees with the gold step on 7.7% of windows, so a model trained on always-correct steps is out of distribution and fires on 93%.
7. Protocol: online causal, clip_len 4s, stride 4s, fps 1; trailing <stride s per clip unsampled (floor tiling).
8. 69 nonlocalizable (start=-1, mostly Missing-Step) segments mark NO window and are unrecallable by windowed metrics; 1 'Other'-only step(s) dropped from typed metrics. missing_step is structurally unreachable -> macro_f1_excl_missing_step.
9. CIs: clip-level bootstrap, n=1000, seed=0, percentile 95%. Windows within a clip are correlated; clips are the resampling unit.
10. EDS must be read jointly with precision: a system that fires early and often scores high EDS regardless of accuracy.
11. Localization uses min_consecutive persistence from each row's decision config; binary F1 gates on smoothed score>0 — different decision rules by design.
12. auc_raw ranks by the raw (ungated) per-window confidence — threshold-free, comparable across rows.