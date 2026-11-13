#!/usr/bin/env bash
# Option B: dedicated causal step recogniser (16-s look-back, all gold-lettered
# windows) -> val + test passes with the DAG belief tracker. Content detection
# stays with the saved graph-free/legacy arms; CPU joins them afterwards.
#
#   setsid nohup bash scripts/cc4d_stepreco_chain.sh > /dev/null 2>&1 < /dev/null &
#   (or under tmux; STOP: touch outputs/review_v4/STOP_CHAIN)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/mnt/hdd/Rafin/miniconda3/bin/python
OUT=outputs/review_v4
ADAPTER=outputs/lora_stepreco
LOG=logs/cc4d_stepreco_chain.log
NEED_INFER=${NEED_INFER_MIB:-28000}
NEED_TRAIN=${NEED_TRAIN_MIB:-50000}
CTX=${CONTEXT_SEC:-16}
CFPS=${CONTEXT_FPS:-0.5}
MAXW=${MAX_WINDOWS:-8000}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

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

say "=== step-recogniser chain started (pid $$; context ${CTX}s @ ${CFPS} fps, max_windows ${MAXW}) ==="

# S1: train the recogniser on all gold-lettered windows (random cap), 1 epoch
if [ -e "$ADAPTER/adapter_model.safetensors" ]; then say "SKIP S1 — adapter present"; else
    check_stop; wait_for_gpu "$NEED_TRAIN"; say "S1: start"
    attempt "S1" $PY scripts/cc4d_lora_train.py --output_dir "$ADAPTER" --no_wandb --resume \
        --prompt_variant stepreco --epochs 1 --neg_per_pos 0 --max_windows "$MAXW" \
        --context_sec "$CTX" --context_fps "$CFPS" --step_loss_weight 1.0 --gradient_checkpointing \
        && say "S1: done" || { say "S1 FAILED"; exit 1; }
fi

# S2/S3: val and test passes, belief tracker, legacy evaluator (records are step-only)
# val: a stratified 31-recording subset (filter/fusion/decision selection only); test: full.
for split in val test; do
    preds="$OUT/stepreco_${split}_preds.json"
    if [ -e "$preds" ]; then say "SKIP S-$split — present"; continue; fi
    extra=""; [ "$split" = val ] && extra="--limit 31 --stratify"
    check_stop; wait_for_gpu "$NEED_INFER"; say "S-$split: start"
    attempt "S-$split" $PY scripts/cc4d_collect_preds.py --split "$split" --stride 4 --resume $extra \
        --prompt_variant stepreco --tracker belief --max_new_tokens 16 \
        --context_sec "$CTX" --context_fps "$CFPS" --lora_adapter "$ADAPTER" \
        --evaluation_protocol legacy_v1 --save_preds "$preds" --out "$OUT/stepreco_${split}_eval.json" \
        && say "S-$split: done" || { say "S-$split FAILED"; exit 1; }
done
say "=== step-recogniser chain finished ==="
