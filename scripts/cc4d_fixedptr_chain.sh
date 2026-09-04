#!/usr/bin/env bash
# "The graph, delivered properly": retrain the graph arm with the corrected step
# tracking, then evaluate it on the full test split and re-gate with R4's frozen
# decision layer.
#
# Only the graph arm needs re-running. R5 (graph off at inference) and R6
# (graph-free throughout) contain no graph in the prompt at all, so no step-
# assignment rule can affect them -- they stay valid as controls, and the new
# row is compared against them directly.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/mnt/hdd/Rafin/miniconda3/bin/python
OUT=outputs/decision_v2
ADAPTER=outputs/lora_cc4d_8class_fixedptr
PARAMS=$OUT/decision_params_lora.json
LOG=logs/cc4d_fixedptr_chain.log
NEED_INFER=${NEED_INFER_MIB:-28000}
NEED_TRAIN=${NEED_TRAIN_MIB:-66000}

mkdir -p "$OUT" logs
exec >> "$LOG" 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
check_stop() { [ -f "$OUT/STOP_CHAIN" ] && { say "STOP_CHAIN — aborting"; exit 0; }; return 0; }

wait_for_gpu() {
    local ok=0 q used total util free min_free=${1:-$NEED_INFER}
    say "waiting for GPU (need >=${min_free} MiB free)"
    while [ $ok -lt 3 ]; do
        check_stop
        q=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
        read -r used total util <<< "$q"
        if ! [[ "${used:-}" =~ ^[0-9]+$ && "${total:-}" =~ ^[0-9]+$ ]]; then
            ok=0; say "gpu unavailable — waiting"; sleep 60; continue
        fi
        free=$(( total - used ))
        if [ "$free" -ge "$min_free" ]; then ok=$(( ok + 1 )); say "gpu ok ($ok/3): free=${free}MiB";
        else ok=0; say "gpu busy: free=${free}MiB"; fi
        sleep 20
    done
}

attempt() {
    local label="$1"; shift
    local tries=0 max=${STAGE_MAX_TRIES:-8}
    while [ "$tries" -lt "$max" ]; do
        tries=$(( tries + 1 ))
        if "$@"; then return 0; fi
        say "$label: attempt $tries/$max failed (co-tenant OOM?) — backing off"
        check_stop; sleep 180; wait_for_gpu "$NEED_INFER"
    done
    say "$label: giving up after $max attempts"; return 1
}

say "=== fixed-pointer chain started (pid $$) ==="

# ---- T1: retrain the graph arm with gold step conditioning ----
if [ -e "$ADAPTER/adapter_model.safetensors" ]; then
    say "SKIP T1 — adapter already present"
else
    check_stop; wait_for_gpu "$NEED_TRAIN"
    say "T1: retrain graph arm with gold-step conditioning"
    if ! $PY scripts/cc4d_lora_train.py --output_dir "$ADAPTER" --no_wandb --resume; then
        say "T1 FAILED — aborting"; exit 1
    fi
    say "T1 done: adapter at $ADAPTER"
fi

# ---- T2: full test pass with the rate-limited pointer ----
if [ -e "$OUT/row_lora_fixedptr_test.json" ]; then
    say "SKIP T2 — row already present"
else
    check_stop; wait_for_gpu "$NEED_INFER"
    say "T2: full test pass, graph prompt + fixed pointer"
    if attempt "T2" $PY scripts/cc4d_collect_preds.py --split test --stride 4 \
            --prompt_variant graph --lora_adapter "$ADAPTER" --resume \
            --save_preds "$OUT/lora_fixedptr_test_preds.json" \
            --out "$OUT/lora_fixedptr_test_eval.json"; then
        say "T2 done — re-gating with R4's frozen decision layer"
        tau=$($PY -c "import json;print(json.load(open('$PARAMS'))['tau'])")
        k=$($PY -c "import json;print(json.load(open('$PARAMS'))['smooth_k'])")
        mc=$($PY -c "import json;print(json.load(open('$PARAMS'))['min_consecutive'])")
        gate=$($PY -c "import json;print(json.load(open('$PARAMS'))['gate_mode'])")
        $PY scripts/cc4d_decision_eval.py --preds "$OUT/lora_fixedptr_test_preds.json" \
            --split test --tau "$tau" --smooth_k "$k" --min_consecutive "$mc" \
            --gate_mode "$gate" --out "$OUT/row_lora_fixedptr_test.json"
        $PY scripts/cc4d_bootstrap_compare.py --preds "$OUT/lora_fixedptr_test_preds.json" \
            --params "$PARAMS" --split test --label lora_fixedptr
        $PY scripts/cc4d_bootstrap_compare.py --preds "$OUT/lora_fixedptr_test_preds.json" \
            --params "$PARAMS" --preds_b "$OUT/lora_nogr_trained_test_preds.json" \
            --params_b "$PARAMS" --split test --label fixedptr_vs_nogrtrained
        say "=== CUT LINE E reached: fixed-pointer graph arm complete ==="
    else
        say "T2 FAILED"; exit 1
    fi
fi
say "=== fixed-pointer chain finished ==="
