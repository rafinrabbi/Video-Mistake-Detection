#!/usr/bin/env bash
# Graph-ablation GPU chain: waits for the GPU to free up, then runs
#   S1  smoke   nograph prompt, 1 clip                      (~5 min)
#   G1  graph-OFF inference, existing adapter, full test    (~8-10 h)
#   G2a no-graph LoRA retrain (identical hyperparams)       (~3 h)
#   G2b no-graph adapter, full test                         (~8-10 h)
# After each test pass: CPU re-gate with R4's frozen decision params,
# bootstrap CIs, rebuild the ablation table.
#
# Cancel: touch outputs/decision_v2/STOP_CHAIN (checked between stages),
# or kill the process group.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=/mnt/hdd/Rafin/miniconda3/bin/python
OUT=outputs/decision_v2
ADAPTER=outputs/lora_cc4d_8class
ADAPTER_NG=outputs/lora_cc4d_8class_nograph
PARAMS=$OUT/decision_params_lora.json
LOG=logs/cc4d_graph_ablation_chain.log

# Per-stage VRAM requirement, MiB. Inference holds the bf16 weights (15.45 GiB,
# measured) plus vision activations and KV cache. Training additionally holds
# un-checkpointed activations for a ~1850-token multimodal sequence, which is what
# makes it roughly 3x the ask. Override with NEED_INFER_MIB / NEED_TRAIN_MIB.
NEED_INFER=${NEED_INFER_MIB:-28000}
NEED_TRAIN=${NEED_TRAIN_MIB:-72000}   # measured: 67.5 GiB reserved at peak, plus context

mkdir -p "$OUT" logs
exec >> "$LOG" 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

# Skip a stage whose output already exists. Without this, a chain that dies
# part-way (as one did when its script was edited mid-run) has to be restarted
# from the beginning, and G2a in particular would RETRAIN FROM SCRATCH: the
# training state file is deleted on success, so --resume finds nothing.
# Retry a stage that fails. On a card shared with several other users an OOM is
# a transient condition, not a reason to discard a multi-day queue: the memory
# check and the model load are seconds apart, so a co-tenant can take the memory
# in between no matter how high the threshold is. Retries are cheap and safe
# because every stage passes --resume: a repeat continues from the recordings
# already on disk rather than restarting the pass.
attempt() {
    local label="$1"; shift
    local tries=0 max=${STAGE_MAX_TRIES:-8}
    while [ "$tries" -lt "$max" ]; do
        tries=$(( tries + 1 ))
        if "$@"; then return 0; fi
        say "$label: attempt $tries/$max failed (co-tenant OOM?) — backing off, then retrying with --resume"
        check_stop
        sleep 180
        wait_for_gpu "$NEED_INFER"
    done
    say "$label: giving up after $max attempts"
    return 1
}

have() {
    if [ -e "$1" ]; then say "SKIP $2 — output already present ($1)"; return 0; fi
    return 1
}

check_stop() {
    if [ -f "$OUT/STOP_CHAIN" ]; then
        say "STOP_CHAIN sentinel found — aborting chain."
        exit 0
    fi
}

wait_for_gpu() {
    # Gate on FREE MEMORY, which is the real constraint: inference needs ~22 GiB,
    # training ~40-60 GiB. Utilisation is a courtesy check only -- a busy card
    # with 85 GiB free hosts this run fine, and gating hard on utilisation left
    # the chain parked for hours behind a job using 13 GiB. Both thresholds are
    # overridable:  GPU_MIN_FREE_MIB (default 40000) and GPU_MAX_UTIL (default 90).
    # Set GPU_MAX_UTIL=30 to restore the old exclusive-card behaviour.
    #
    # A missing or unhealthy GPU stays a WAIT condition, not an error: the card
    # can fall off the bus, and an unguarded read of that output left util unset,
    # which under 'set -u' killed the chain and lost a queued multi-day run.
    # $1 = MiB of free VRAM this stage needs; inference and training differ ~3x.
    local ok=0 query
    local min_free=${1:-${GPU_MIN_FREE_MIB:-40000}}
    local max_util=${GPU_MAX_UTIL:-101}
    say "waiting for GPU (need >=${min_free} MiB free, <${max_util}% util, 3 consecutive checks)"
    while [ $ok -lt 3 ]; do
        check_stop
        query=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
        read -r used total util <<< "$query"
        if ! [[ "${used:-}" =~ ^[0-9]+$ && "${total:-}" =~ ^[0-9]+$ && "${util:-}" =~ ^[0-9]+$ ]]; then
            ok=0
            say "gpu unavailable (nvidia-smi returned no usable row) -- still waiting"
            sleep 60
            continue
        fi
        free=$(( total - used ))
        if [ "$free" -ge "$min_free" ] && [ "$util" -lt "$max_util" ]; then
            ok=$(( ok + 1 ))
            say "gpu check ok ($ok/3): free=${free}MiB util=${util}%"
        else
            ok=0
            say "gpu busy: free=${free}MiB util=${util}% -- waiting"
        fi
        sleep 20
    done
    say "GPU free — proceeding"
}

regate() {
    # $1 = preds json, $2 = row basename, $3 = bootstrap label
    local tau k mc gate
    tau=$($PY -c "import json;print(json.load(open('$PARAMS'))['tau'])")
    k=$($PY -c "import json;print(json.load(open('$PARAMS'))['smooth_k'])")
    mc=$($PY -c "import json;print(json.load(open('$PARAMS'))['min_consecutive'])")
    gate=$($PY -c "import json;print(json.load(open('$PARAMS'))['gate_mode'])")
    $PY scripts/cc4d_decision_eval.py --preds "$1" --split test \
        --tau "$tau" --smooth_k "$k" --min_consecutive "$mc" --gate_mode "$gate" \
        --out "$OUT/$2"
    $PY scripts/cc4d_bootstrap_compare.py --preds "$1" --params "$PARAMS" \
        --split test --label "$3"
    $PY scripts/cc4d_build_ablation.py --manifest "$OUT/ablation_manifest.json" \
        > /dev/null && say "ablation table rebuilt"
}

say "=== chain started (pid $$) ==="

# ---- S1: smoke ----
wait_for_gpu "$NEED_INFER"
check_stop
say "S1 smoke: nograph prompt, 1 clip, stride 8"
if ! $PY scripts/cc4d_collect_preds.py --limit 1 --stride 8 \
        --prompt_variant nograph --lora_adapter "$ADAPTER" \
        --save_preds "$OUT/smoke_nograph_preds.json"; then
    say "SMOKE FAILED — aborting chain"; exit 1
fi
say "S1 smoke OK"

# ---- G1: graph-off inference, full test ----
if ! have "$OUT/row_lora_nograph_test.json" "G1"; then
check_stop
wait_for_gpu "$NEED_INFER"
say "G1: graph-OFF inference (existing adapter), full test"
if $PY scripts/cc4d_collect_preds.py --split test --stride 4 \
        --prompt_variant nograph --lora_adapter "$ADAPTER" --resume \
        --save_preds "$OUT/lora_nograph_test_preds.json" \
        --out "$OUT/lora_nograph_test_eval.json"; then
    say "G1 done — re-gating with R4 params"
    regate "$OUT/lora_nograph_test_preds.json" row_lora_nograph_test.json lora_nograph
    say "=== CUT LINE A reached: inference graph ablation complete ==="

    # QF: 3 clips with the text fields logged, for the qualitative figure.
    # Cheap (~30 min) and placed here so the figure is unblocked early rather
    # than after another day of inference.
    check_stop
    wait_for_gpu "$NEED_INFER"
    say "QF: qualitative collection, 3 clips, --log_text"
    if $PY scripts/cc4d_collect_preds.py --split test --limit 3 --stride 4 \
            --lora_adapter "$ADAPTER" --log_text --resume \
            --save_preds "$OUT/qualitative_preds.json"; then
        say "QF done -> $OUT/qualitative_preds.json"
    else
        say "QF FAILED (non-fatal) — continuing"
    fi
else
    say "G1 FAILED — aborting chain"; exit 1
fi
fi

# ---- G2a: no-graph LoRA retrain ----
if ! have "$ADAPTER_NG/adapter_model.safetensors" "G2a"; then
check_stop
wait_for_gpu "$NEED_TRAIN"
say "G2a: retrain LoRA with nograph prompts (identical hyperparams)"
if ! $PY scripts/cc4d_lora_train.py --output_dir "$ADAPTER_NG" \
        --prompt_variant nograph --no_wandb --resume; then
    say "G2a FAILED — aborting chain"; exit 1
fi
say "G2a done: adapter at $ADAPTER_NG"
fi

# ---- G2b: no-graph adapter, full test ----
if ! have "$OUT/row_lora_nogr_trained_test.json" "G2b"; then
check_stop
wait_for_gpu "$NEED_INFER"
say "G2b: no-graph-trained adapter, graph-OFF prompt, full test"
if attempt "G2b" $PY scripts/cc4d_collect_preds.py --split test --stride 4 \
        --prompt_variant nograph --lora_adapter "$ADAPTER_NG" --resume \
        --save_preds "$OUT/lora_nogr_trained_test_preds.json" \
        --out "$OUT/lora_nogr_trained_test_eval.json"; then
    say "G2b done — re-gating with R4 params"
    regate "$OUT/lora_nogr_trained_test_preds.json" row_lora_nogr_trained_test.json lora_nogr_trained
    say "=== CUT LINE B reached: full graph ablation complete ==="
else
    say "G2b FAILED"; exit 1
fi
fi

# ---- G3: wrong-graph control (R7) ----
# Structure present, content wrong: every task is conditioned on a DIFFERENT
# task's graph (fixpoint-free derangement, fixed seed). Separates "the plan's
# content helps" from "any structured scaffold helps". Uses the ORIGINAL
# graph-trained adapter and the graph prompt -- only the graph content changes.
if ! have "$OUT/row_lora_wronggraph_test.json" "G3"; then
check_stop
wait_for_gpu "$NEED_INFER"
say "G3: wrong-graph control (deranged graphs, seed 0), full test"
if attempt "G3" $PY scripts/cc4d_collect_preds.py --split test --stride 4 \
        --prompt_variant graph --lora_adapter "$ADAPTER" \
        --graph_permute_seed 0 --resume \
        --save_preds "$OUT/lora_wronggraph_test_preds.json" \
        --out "$OUT/lora_wronggraph_test_eval.json"; then
    say "G3 done — re-gating with R4 params"
    regate "$OUT/lora_wronggraph_test_preds.json" row_lora_wronggraph_test.json lora_wronggraph
    say "=== CUT LINE C reached: wrong-graph control complete ==="
else
    say "G3 FAILED"; exit 1
fi
fi

# ---- G4: static-dump control (R10) ----
# Same graph CONTENT as the main arm, but the full step list is dumped and the
# model is not told where in it the person should be. This is the only row that
# isolates the WALKED POINTER from merely having the plan, and it is what the
# manuscript needs before it can claim the walk as a mechanism rather than
# describing it as a design choice. Last because it is the most droppable.
if ! have "$OUT/row_lora_staticdump_test.json" "G4"; then
check_stop
wait_for_gpu "$NEED_INFER"
say "G4: static-dump control (position removed), full test"
if attempt "G4" $PY scripts/cc4d_collect_preds.py --split test --stride 4 \
        --prompt_variant staticdump --lora_adapter "$ADAPTER" --resume \
        --save_preds "$OUT/lora_staticdump_test_preds.json" \
        --out "$OUT/lora_staticdump_test_eval.json"; then
    say "G4 done — re-gating with R4 params"
    regate "$OUT/lora_staticdump_test_preds.json" row_lora_staticdump_test.json lora_staticdump
    say "=== CUT LINE D reached: walked-pointer control complete ==="
else
    say "G4 FAILED"; exit 1
fi
fi

say "=== chain finished ==="
