#!/usr/bin/env bash
# CPU replay for the pre-registered confirmation.
# Idempotent: each matched run / aggregate is skipped when its output exists,
# and skipped (not failed) while its GPU inputs are still missing.
#
#   bash scripts/cc4d_confirm_post.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/mnt/hdd/Rafin/miniconda3/bin/python
C=outputs/confirm_v1
STATUS="confirmatory (pre-registered 2026-10-20); seed 42 TEST previously examined"
say() { echo "[$(date '+%F %T')] $*"; }

declare -A STEP_VAL=([s42]=$C/stepreco_s42_val_preds.json [s1]=$C/stepreco_s1_val_preds.json
                     [s2]=$C/stepreco_s2_val_preds.json)
declare -A STEP_TEST=([s42]=outputs/review_v4/stepreco_test_preds.json [s1]=$C/stepreco_s1_test_preds.json
                      [s2]=$C/stepreco_s2_test_preds.json)
declare -A CONTENT_VAL=([r4]=outputs/full_eval_8class/lora_val_preds.json [r6]=$C/r6_val_preds.json
                        [mistsense]=outputs/decision_v2/mistsense_val_preds.json)
declare -A CONTENT_TEST=([r4]=outputs/full_eval_8class/lora_test_preds.json
                         [r6]=outputs/decision_v2/lora_nogr_trained_test_preds.json
                         [mistsense]=outputs/decision_v2/mistsense_test_preds.json)

matched() {  # out_dir, step_val, step_test, content_val, content_test, title, extra...
    local out="$1" sv="$2" st="$3" cv="$4" ct="$5" title="$6"; shift 6
    if [ -e "$out/summary.json" ]; then return 0; fi
    for f in "$sv" "$st" "$cv" "$ct"; do [ -e "$f" ] || { say "WAIT $out — missing $f"; return 1; }; done
    rm -rf "$out"
    say "RUN $out"
    $PY -u scripts/cc4d_matched_graph_experiment.py --out_dir "$out" --stepreco_val "$sv" --stepreco_test "$st" \
        --content_val "$cv" --content_test "$ct" --status "$STATUS" --title "$title" "$@" > "$out.log" 2>&1 \
        || { say "FAILED $out (see $out.log)"; return 1; }
}

for content in r4 r6 mistsense; do
    runs=()
    for seed in s42 s1 s2; do
        out=$C/matched_${content}_${seed}
        matched "$out" "${STEP_VAL[$seed]}" "${STEP_TEST[$seed]}" "${CONTENT_VAL[$content]}" \
            "${CONTENT_TEST[$content]}" "Matched topology, content=$content, recogniser $seed" && runs+=(--run "$out")
    done
    agg=$C/aggregate_${content}.json
    primary=""; [ "$content" = r4 ] && primary=--primary
    if [ ${#runs[@]} -eq 6 ] && [ ! -e "$agg" ]; then
        say "AGGREGATE $content"
        $PY scripts/cc4d_confirm_aggregate.py --label "$content" "${runs[@]}" --out "$agg" $primary \
            > "$C/aggregate_${content}.log" 2>&1 || say "FAILED aggregate $content"
    fi
done

# Amendment 2 / S11: the 3x-data recogniser, scored exactly like the other seeds (R4 content).
out=$C/matched_r4_24k
if matched "$out" $C/stepreco_24k_val_preds.json $C/stepreco_24k_test_preds.json "${CONTENT_VAL[r4]}" \
        "${CONTENT_TEST[r4]}" "Matched topology, content=r4, recogniser 24k (amendment 2)" \
        && [ ! -e $C/aggregate_r24k.json ]; then
    say "AGGREGATE r24k"
    $PY scripts/cc4d_confirm_aggregate.py --label r24k --run "$out" --out $C/aggregate_r24k.json \
        > $C/aggregate_r24k.log 2>&1 || say "FAILED aggregate r24k"
fi

out=$C/matched_holdout
if matched "$out" $C/stepreco_holdout_val_preds.json $C/stepreco_holdout_test_preds.json \
        $C/nograph_holdout_val_preds.json outputs/review_v4/nograph_holdout_test_preds.json \
        "Matched topology, unseen recipes" --recipe_holdout data/splits/recipe_holdout_v1.json \
        --step_seconds data/step_seconds_holdout_v1.json && [ ! -e $C/aggregate_holdout.json ]; then
    say "AGGREGATE holdout"
    $PY scripts/cc4d_confirm_aggregate.py --label holdout --run "$out" --out $C/aggregate_holdout.json \
        --recipe_holdout data/splits/recipe_holdout_v1.json > $C/aggregate_holdout.log 2>&1 \
        || say "FAILED aggregate holdout"
fi
say "post pass complete"
