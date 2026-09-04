#!/usr/bin/env python
r"""Generate every results table in the manuscript directly from the result JSONs.

Numbers in the paper are never typed by hand: run this after any new eval row
lands and the .tex tables update in place.

    python paper/gen_tables.py

Rows that have not been produced yet are emitted as red \ph{} placeholders so a
missing experiment is visible in the compiled PDF instead of silently absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DV2 = ROOT / "outputs" / "decision_v2"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    ("no_mistake", "no\\_mistake"),
    ("order_error", "order"),
    ("technique_error", "technique"),
    ("preparation_error", "preparation"),
    ("measurement_error", "measurement"),
    ("timing_error", "timing"),
    ("temperature_error", "temperature"),
    ("missing_step", "missing\\_step"),
]


def emit(path, lines):
    """Write table rows plus the closing rule.

    ``\\input`` appends a space token at end of file, which lands inside the
    alignment and makes a following ``\\bottomrule`` a misplaced ``\\noalign``.
    Emitting the rule here keeps it on the correct side of that boundary."""
    (OUT / path).write_text("\n".join(lines) + "\n\\bottomrule%")

def load(name):
    p = DV2 / name
    return json.loads(p.read_text()) if p.exists() else None


def ci(boot, key):
    """'0.482 [0.392, 0.558]' or an em dash when the bootstrap is absent."""
    if not boot or key not in boot.get("ci95", {}):
        return "--"
    lo, hi = boot["ci95"][key]
    return f"{boot['point'][key]:.3f} \\ci{{{lo:.3f}}}{{{hi:.3f}}}"


def num(row, key, fmt="{:.3f}"):
    if row is None:
        return "--"
    v = row["metrics"].get(key)
    return "--" if v is None else fmt.format(v)


# --------------------------------------------------------------------------
# Table 1: main results, online causal, full test split
# --------------------------------------------------------------------------
MAIN_ROWS = [
    # (label, row file, bootstrap file, pending?)
    ("Zero-shot Qwen2.5-VL, graph prompt",
     "row_zs_tuned_test.json", "bootstrap_zs_tuned.json", None),
    ("\\ourmodel{} (graph prompt + LoRA)",
     "row_lora_tuned_test.json", "bootstrap_lora_tuned.json", None),
    ("\\midrule \\ourmodel{} w/o graph at inference (R5)",
     "row_lora_nograph_test.json", "bootstrap_lora_nograph.json", None),
    ("\\ourmodel{} trained \\emph{and} tested graph-free (R6)",
     "row_lora_nogr_trained_test.json", "bootstrap_lora_nogr_trained.json", None),
    ("\\ourmodel{} with corrected step pointer (R11)",
     "row_lora_fixedptr_test.json", "bootstrap_lora_fixedptr.json", None),
    ("\\midrule MistSense~\\citep{patsch2025mistsense}, protocol-matched",
     "row_mistsense_tuned_test.json", "bootstrap_mistsense_tuned.json", None),
]

PENDING_MAIN = [
]


def table_main():
    lines = []
    for label, rowf, bootf, _ in MAIN_ROWS:
        row, boot = load(rowf), load(bootf)
        lines.append(
            f"{label} & {ci(boot,'binary_f1')} & {num(row,'precision')} & "
            f"{num(row,'recall')} & {ci(boot,'macro_f1')} & "
            f"{num(row,'macro_f1_excl_missing_step')} & {ci(boot,'auc_raw')} & "
            f"{num(row,'ece')} & "
            f"{ci(boot,'eds')} & {num(row,'start_mae','{:.1f}')} \\\\"
        )
    # The trivial row is re-gated through the identical decision layer, so it is
    # a real row of this table rather than a number quoted beside it. ECE is
    # left blank: a constant score has no calibration curve to speak of.
    af = load("row_alwaysfire_test.json")
    if af:
        a = af["metrics"]
        lines.append("\\midrule Fire on every window (trivial) & "
                     f"{a['binary_f1']:.3f} & {a['precision']:.3f} & {a['recall']:.3f} & "
                     f"{a['macro_f1']:.3f} & {a['macro_f1_excl_missing_step']:.3f} & "
                     f"{a['auc_raw']:.3f} & -- & {a['eds']:.3f} & {a['start_mae']:.1f} \\\\")
    if PENDING_MAIN:
        lines.append("\\midrule")
    for label, tag in PENDING_MAIN:
        lines.append(f"{label} & \\ph{{{tag}}} & \\phd & \\phd & \\phd & \\phd & \\phd & \\phd & \\phd & \\phd \\\\")
    emit("main_results.tex", lines)


# --------------------------------------------------------------------------
# Table 2: decision-layer / confidence-source ablation (all CPU re-gates)
# --------------------------------------------------------------------------
def table_decision():
    rows = [
        ("Default gate ($\\tau{=}0.5$, AND, $k{=}3$)", "row_lora_default_test.json",
         "bootstrap_lora_default.json"),
        ("Validation-tuned gate (ours)", "row_lora_tuned_test.json",
         "bootstrap_lora_tuned.json"),
    ]
    lines = []
    for label, rowf, bootf in rows:
        row, boot = load(rowf), load(bootf)
        lines.append(
            f"{label} & {ci(boot,'binary_f1')} & {num(row,'precision')} & "
            f"{num(row,'recall')} & {num(row,'eds')} & {num(row,'start_mae','{:.1f}')} \\\\"
        )
    # deviation-flag-only row: recomputed on CPU from the saved predictions,
    # equivalent to gating on the (constant) verbalized confidence.
    dev = load("row_lora_devonly_test.json")
    if dev:
        lines.insert(0,
            f"Deviation flag only (no confidence gate) & {num(dev,'binary_f1')} & "
            f"{num(dev,'precision')} & {num(dev,'recall')} & {num(dev,'eds')} & "
            f"{num(dev,'start_mae','{:.1f}')} \\\\")
    else:
        lines.insert(0, "Deviation flag only (no confidence gate) & \\ph{RUN-DEVONLY} & \\phd & \\phd & \\phd & \\phd \\\\")
    emit("decision_ablation.tex", lines)


# --------------------------------------------------------------------------
# Table 3: per-class F1 with support
# --------------------------------------------------------------------------
def table_perclass():
    zs, lora = load("row_zs_tuned_test.json"), load("row_lora_tuned_test.json")
    if not lora:
        return
    lines = []
    for key, disp in CLASSES:
        sup = lora["metrics"]["support"].get(key, 0)
        z = zs["metrics"]["per_class_f1"].get(key, 0.0) if zs else float("nan")
        l = lora["metrics"]["per_class_f1"].get(key, 0.0)
        lines.append(f"\\texttt{{{disp}}} & {sup:,} & {z:.3f} & {l:.3f} \\\\")
    emit("per_class.tex", lines)


# --------------------------------------------------------------------------
# Appendix table: the complete experiment register
#
# Every row ever produced under the frozen decision layer, including the ones
# the main table has no room for (the default-gate guard rows) and the two
# controls that were cancelled. Built from the same ablation.json the
# repository's own table is built from, so the paper and the artefact cannot
# disagree.
# --------------------------------------------------------------------------
REGISTER = [
    ("R1",  "Zero-shot, graph prompt, default gate",        "R1 "),
    ("R2",  "Zero-shot, graph prompt, val-tuned gate",      "R2 "),
    ("R3",  "LoRA + graph, default gate (guard row)",       "R3 "),
    ("R4",  "LoRA + graph, val-tuned gate \\textbf{(main)}", "R4 "),
    ("R5",  "LoRA + graph, graph stripped at inference",    "R5 "),
    ("R6",  "LoRA graph-free, trained and tested",          "R6 "),
    ("R11", "LoRA + graph, corrected step pointer",         "R11 "),
    ("R9",  "MistSense, re-tiled, val-tuned gate",          "R9 "),
    ("R12", "Fire on every window (trivial)",               "R12 "),
]


def table_register():
    abl = ROOT / "outputs" / "decision_v2" / "ablation.json"
    if not abl.exists():
        return
    data = json.loads(abl.read_text())
    by_prefix = {r["row"]: r for r in data["rows"]}

    def find(pref):
        for name, r in by_prefix.items():
            if name.startswith(pref):
                return r
        return None

    lines = []
    for tag, desc, pref in REGISTER:
        r = find(pref)
        if r is None:
            continue
        def g(k, fmt="{:.3f}"):
            v = r.get(k)
            return "--" if v is None else fmt.format(v)
        lines.append(
            f"{tag} & {desc} & {g('binary_f1')} & {g('macro_f1')} & "
            f"{g('auc_raw')} & {g('precision')} & {g('recall')} & "
            f"{g('eds')} & {g('start_mae','{:.1f}')} \\\\")
    lines.append("\\midrule")
    lines.append("R7 & Deranged (wrong-recipe) graph & \\multicolumn{7}{c}{"
                 "\\emph{pre-registered, cancelled unrun (Table~\\ref{tab:verdict})}} \\\\")
    lines.append("R10 & Static dump, position removed & \\multicolumn{7}{c}{"
                 "\\emph{pre-registered, cancelled unrun}} \\\\")
    emit("register.tex", lines)


# --------------------------------------------------------------------------
# Macros: every scalar quoted in the prose, so the text cannot drift either
# --------------------------------------------------------------------------
def macros():
    lora, zs = load("row_lora_tuned_test.json"), load("row_zs_tuned_test.json")
    cmp_ = load("bootstrap_lora_vs_zs.json")
    params = load("decision_params_lora.json")
    base = load("row_mistsense_tuned_test.json")
    bparams = load("decision_params_mistsense.json")
    bcmp = load("bootstrap_lora_vs_mistsense.json")
    ng = load("row_lora_nograph_test.json")
    ngc = load("bootstrap_lora_vs_nograph.json")
    ngd = ngc["delta_a_minus_b"] if ngc else None
    sat = load("pointer_saturation.json")
    tb = load("row_trivial_baselines.json")
    afc = load("bootstrap_r4_vs_alwaysfire.json")
    fx = load("row_lora_fixedptr_test.json")
    fxc = load("bootstrap_fixedptr_vs_r4.json")
    mo = load("matched_onset.json")
    cs = load("causal_smoothing.json")
    nt = load("row_lora_nogr_trained_test.json")
    ntc = load("bootstrap_lora_vs_nogrtrained.json")
    ntd = ntc["delta_a_minus_b"] if ntc else None
    bd = bcmp["delta_a_minus_b"]["binary_f1"] if bcmp else None
    m = lora["metrics"]
    d = cmp_["delta_a_minus_b"]["binary_f1"]
    defs = {
        "resF": f"{m['binary_f1']:.3f}",
        "resZS": f"{zs['metrics']['binary_f1']:.3f}",
        "resDelta": f"{d['point']:.3f}",
        "resDeltaLo": f"{d['ci95'][0]:.3f}",
        "resDeltaHi": f"{d['ci95'][1]:.3f}",
        "resAUC": f"{m['auc_raw']:.3f}",
        "resAUCzs": f"{zs['metrics']['auc_raw']:.3f}",
        "resMacro": f"{m['macro_f1']:.3f}",
        "resMacroX": f"{m['macro_f1_excl_missing_step']:.3f}",
        "resP": f"{m['precision']:.3f}",
        "resR": f"{m['recall']:.3f}",
        "resEDS": f"{m['eds']:.3f}",
        "resMAE": f"{m['start_mae']:.1f}",
        "resECE": f"{m['ece']:.3f}",
        "resBrier": f"{m['brier']:.3f}",
        "resDet": f"{m['n_loc_detected']}",
        "resLoc": f"{m['n_loc_events']}",
        "resWindows": f"{m['n_windows']:,}",
        "resClips": f"{m['n_clips']}",
        "resPmc": f"{cmp_['mcnemar']['p_two_sided']:.1e}".replace("e-0", "\\times 10^{-") + "}",
        "baseF": f"{base['metrics']['binary_f1']:.3f}" if base else "\\phd",
        "baseP": f"{base['metrics']['precision']:.3f}" if base else "\\phd",
        "baseR": f"{base['metrics']['recall']:.3f}" if base else "\\phd",
        "baseAUC": f"{base['metrics']['auc_raw']:.3f}" if base else "\\phd",
        "baseEDS": f"{base['metrics']['eds']:.3f}" if base else "\\phd",
        "baseECE": f"{base['metrics']['ece']:.3f}" if base else "\\phd",
        "baseDet": f"{base['metrics']['n_loc_detected']}" if base else "\\phd",
        "zsECE": f"{zs['metrics']['ece']:.3f}",
        "zsDet": f"{zs['metrics']['n_loc_detected']}",
        "baseTau": f"{bparams['tau']}" if bparams else "\\phd",
        "baseDelta": f"{bd['point']:+.3f}" if bd else "\\phd",
        "baseDeltaLo": f"{bd['ci95'][0]:+.3f}" if bd else "\\phd",
        "baseDeltaHi": f"{bd['ci95'][1]:+.3f}" if bd else "\\phd",
        "ngF": f"{ng['metrics']['binary_f1']:.3f}" if ng else "\\phd",
        "ngAUC": f"{ng['metrics']['auc_raw']:.3f}" if ng else "\\phd",
        "ngMacro": f"{ng['metrics']['macro_f1']:.3f}" if ng else "\\phd",
        "ngEDS": f"{ng['metrics']['eds']:.3f}" if ng else "\\phd",
        "ngMAE": f"{ng['metrics']['start_mae']:.1f}" if ng else "\\phd",
        "ngDelta": f"{ngd['binary_f1']['point']:+.3f}" if ngd else "\\phd",
        "ngDeltaLo": f"{ngd['binary_f1']['ci95'][0]:+.3f}" if ngd else "\\phd",
        "ngDeltaHi": f"{ngd['binary_f1']['ci95'][1]:+.3f}" if ngd else "\\phd",
        "ngEdsDelta": f"{ngd['eds']['point']:+.3f}" if ngd else "\\phd",
        "ngEdsLo": f"{ngd['eds']['ci95'][0]:+.3f}" if ngd else "\\phd",
        "ngEdsHi": f"{ngd['eds']['ci95'][1]:+.3f}" if ngd else "\\phd",
        "satPct": f"{sat['frac_windows_at_final_step']*100:.0f}" if sat else "\\phd",
        "satMed": f"{sat['per_clip_saturated_frac_median']*100:.0f}" if sat else "\\phd",
        "satClips": f"{sat['n_clips_majority_saturated']}" if sat else "\\phd",
        "satReach": f"{sat['reach_final_step_at_frac_median']*100:.0f}" if sat else "\\phd",
        "satTrain": f"{sat['train_frac_windows_at_final_step']*100:.0f}" if sat else "\\phd",
        "satTrainMed": f"{sat['train_per_clip_saturated_frac_median']*100:.0f}" if sat else "\\phd",
        "satN": f"{sat['n_clips_reaching_final_step']}" if sat else "\\phd",
        "ntF": f"{nt['metrics']['binary_f1']:.3f}" if nt else "\\phd",
        "ntAUC": f"{nt['metrics']['auc_raw']:.3f}" if nt else "\\phd",
        "ntMacro": f"{nt['metrics']['macro_f1']:.3f}" if nt else "\\phd",
        "ntEDS": f"{nt['metrics']['eds']:.3f}" if nt else "\\phd",
        "ntMAE": f"{nt['metrics']['start_mae']:.1f}" if nt else "\\phd",
        "ntDet": f"{nt['metrics']['n_loc_detected']}" if nt else "\\phd",
        "ntP": f"{nt['metrics']['precision']:.3f}" if nt else "\\phd",
        "ntR": f"{nt['metrics']['recall']:.3f}" if nt else "\\phd",
        "ntDelta": f"{ntd['binary_f1']['point']:+.3f}" if ntd else "\\phd",
        "ntDeltaLo": f"{ntd['binary_f1']['ci95'][0]:+.3f}" if ntd else "\\phd",
        "ntDeltaHi": f"{ntd['binary_f1']['ci95'][1]:+.3f}" if ntd else "\\phd",
        "ntEdsDelta": f"{ntd['eds']['point']:+.3f}" if ntd else "\\phd",
        "ntEdsLo": f"{ntd['eds']['ci95'][0]:+.3f}" if ntd else "\\phd",
        "ntEdsHi": f"{ntd['eds']['ci95'][1]:+.3f}" if ntd else "\\phd",
        "afF": f"{tb['always_fire']['binary_f1']:.3f}" if tb else "\\phd",
        "afEDS": f"{tb['always_fire']['eds']:.3f}" if tb else "\\phd",
        "afMAE": f"{tb['always_fire']['start_mae']:.1f}" if tb else "\\phd",
        "posRate": f"{tb['positive_rate']*100:.1f}" if tb else "\\phd",
        "afDelta": f"{afc['delta_a_minus_b']['binary_f1']['point']:+.3f}" if afc else "\\phd",
        "afDeltaLo": f"{afc['delta_a_minus_b']['binary_f1']['ci95'][0]:+.3f}" if afc else "\\phd",
        "afDeltaHi": f"{afc['delta_a_minus_b']['binary_f1']['ci95'][1]:+.3f}" if afc else "\\phd",
        "afAucDelta": f"{afc['delta_a_minus_b']['auc_raw']['point']:+.3f}" if afc else "\\phd",
        "afAucLo": f"{afc['delta_a_minus_b']['auc_raw']['ci95'][0]:+.3f}" if afc else "\\phd",
        "afAucHi": f"{afc['delta_a_minus_b']['auc_raw']['ci95'][1]:+.3f}" if afc else "\\phd",
        "fxF": f"{fx['metrics']['binary_f1']:.3f}" if fx else "\\phd",
        "fxP": f"{fx['metrics']['precision']:.3f}" if fx else "\\phd",
        "fxR": f"{fx['metrics']['recall']:.3f}" if fx else "\\phd",
        "fxAUC": f"{fx['metrics']['auc_raw']:.3f}" if fx else "\\phd",
        "fxMacro": f"{fx['metrics']['macro_f1']:.3f}" if fx else "\\phd",
        "fxEDS": f"{fx['metrics']['eds']:.3f}" if fx else "\\phd",
        "fxMacroDelta": f"{fxc['delta_a_minus_b']['macro_f1']['point']:+.3f}" if fxc else "\\phd",
        "fxMacroLo": f"{fxc['delta_a_minus_b']['macro_f1']['ci95'][0]:+.3f}" if fxc else "\\phd",
        "fxMacroHi": f"{fxc['delta_a_minus_b']['macro_f1']['ci95'][1]:+.3f}" if fxc else "\\phd",
        "moMed": f"{mo['R4_vs_R6']['median_diff_s']:+.0f}" if mo else "\\phd",
        "moLo": f"{mo['R4_vs_R6']['ci95'][0]:+.0f}" if mo else "\\phd",
        "moHi": f"{mo['R4_vs_R6']['ci95'][1]:+.0f}" if mo else "\\phd",
        "moN": f"{mo['R4_vs_R6']['n']}" if mo else "\\phd",
        "moEarlier": f"{mo['R4_vs_R6']['a_earlier']}" if mo else "\\phd",
        "moLater": f"{mo['R4_vs_R6']['a_later']}" if mo else "\\phd",
        # Strictly causal (trailing) decision layer -- the centred median
        # filter's 8 s lookahead, priced. See scripts/cc4d_causal_smoothing.py.
        "csF": f"{cs['arms']['R4']['trailing']['binary_f1']:.3f}" if cs else "\\phd",
        "csDelta": f"{cs['arms']['R4']['delta']['binary_f1']:+.3f}" if cs else "\\phd",
        "csEDS": f"{cs['arms']['R4']['trailing']['eds']:.3f}" if cs else "\\phd",
        "csEdsDelta": f"{cs['arms']['R4']['delta']['eds']:+.3f}" if cs else "\\phd",
        "csLook": f"{int(params['smooth_k'] // 2 * 4)}" if params else "\\phd",
        "csNgF": f"{cs['arms']['R5']['trailing']['binary_f1']:.3f}" if cs else "\\phd",
        "csNtF": f"{cs['arms']['R6']['trailing']['binary_f1']:.3f}" if cs else "\\phd",
        "csNgEDS": f"{cs['arms']['R5']['trailing']['eds']:.3f}" if cs else "\\phd",
        "csNtEDS": f"{cs['arms']['R6']['trailing']['eds']:.3f}" if cs else "\\phd",
        "gateTau": f"{params['tau']}",
        "gateK": f"{params['smooth_k']}",
        "gateMinC": f"{params['min_consecutive']}",
    }
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items())
    (OUT / "macros.tex").write_text(
        "% AUTO-GENERATED by paper/gen_tables.py -- do not edit by hand.\n" + body + "\n")


if __name__ == "__main__":
    if not DV2.exists():
        sys.exit(f"missing {DV2}")
    table_main(); table_decision(); table_perclass(); table_register(); macros()
    print(f"wrote tables -> {OUT}")
