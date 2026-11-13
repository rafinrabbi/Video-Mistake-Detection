#!/usr/bin/env bash
# Pre-registered confirmation runs.
# Two lanes share the card: lane "infer" runs inference-only passes, lane
# "train" runs the recogniser retrains and their passes. Each stage is skipped
# when its output exists, so either lane can be restarted.
#
#   setsid nohup bash scripts/cc4d_confirm_chain.sh infer > /dev/null 2>&1 < /dev/null &
#   setsid nohup bash scripts/cc4d_confirm_chain.sh train > /dev/null 2>&1 < /dev/null &
#   STOP: touch outputs/confirm_v1/STOP_CHAIN
set -uo pipefail
cd "$(dirname "$0")/.."
LANE=${1:?usage: cc4d_confirm_chain.sh infer|train|reco24k}
PY=/mnt/hdd/Rafin/miniconda3/bin/python
OUT=outputs/confirm_v1
LOG=logs/cc4d_confirm_${LANE}.log
NEED_INFER=${NEED_INFER_MIB:-28000}
NEED_TRAIN=${NEED_TRAIN_MIB:-50000}
HOLDOUT=data/splits/recipe_holdout_v1.json
HOLDOUT_SEC=data/step_seconds_holdout_v1.json
RECO=(--prompt_variant stepreco --tracker belief --max_new_tokens 16 --context_sec 16 --context_fps 0.5
      --evaluation_protocol legacy_v1 --stride 4)
TRAIN=(--prompt_variant stepreco --epochs 1 --neg_per_pos 0 --max_windows 8000 --context_sec 16
       --context_fps 0.5 --step_loss_weight 1.0 --gradient_checkpointing)
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
# 26_19_360p.mp4 trips decord's default EOF retry limit on some passes.
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-40960}

mkdir -p "$OUT" logs
exec >> "$LOG" 2>&1
say() { echo "[$(date '+%F %T')] [$LANE] $*"; }
check_stop() { [ -f "$OUT/STOP_CHAIN" ] && { say "STOP_CHAIN — aborting"; exit 0; }; return 0; }

wait_for_gpu() {
    local ok=0 q used total util free min_free=${1:-$NEED_INFER}
    say "waiting for GPU (need >=${min_free} MiB free)"
    while [ $ok -lt 3 ]; do
        check_stop
        q=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')
        read -r used total util <<< "$q"
        if ! [[ "${used:-}" =~ ^[0-9]+$ && "${total:-}" =~ ^[0-9]+$ ]]; then
            ok=0; sleep 60; continue
        fi
        free=$(( total - used ))
        if [ "$free" -ge "$min_free" ]; then ok=$(( ok + 1 )); else ok=0; fi
        sleep 20
    done
    say "gpu ok: free=${free}MiB"
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

infer() {   # label, preds json, then collect_preds args...
    local label="$1" preds="$2"; shift 2
    if [ -e "$preds" ]; then say "SKIP $label — present"; return 0; fi
    check_stop; wait_for_gpu "$NEED_INFER"; say "$label: start"
    attempt "$label" $PY scripts/cc4d_collect_preds.py --resume --save_preds "$preds" \
        --out "${preds%_preds.json}_eval.json" "$@" \
        && say "$label: done" || { say "$label FAILED"; return 1; }
}

train() {   # label, adapter dir, then lora_train args...
    local label="$1" adapter="$2"; shift 2
    if [ -e "$adapter/adapter_model.safetensors" ]; then say "SKIP $label — adapter present"; return 0; fi
    check_stop; wait_for_gpu "$NEED_TRAIN"; say "$label: start"
    attempt "$label" $PY scripts/cc4d_lora_train.py --output_dir "$adapter" --no_wandb --resume "$@" \
        && say "$label: done" || { say "$label FAILED"; return 1; }
}

say "=== confirmation lane started (pid $$) ==="
if [ "$LANE" = infer ]; then
    infer V42 "$OUT/stepreco_s42_val_preds.json" --split val "${RECO[@]}" \
        --lora_adapter outputs/lora_stepreco || exit 1
    infer VR6 "$OUT/r6_val_preds.json" --split val --stride 4 --prompt_variant nograph \
        --lora_adapter outputs/lora_cc4d_8class_nograph || exit 1
    infer HV "$OUT/nograph_holdout_val_preds.json" --split val --stride 4 --prompt_variant nograph \
        --tracker pointer --lora_adapter outputs/lora_nograph_holdout --recipe_holdout "$HOLDOUT" \
        --step_seconds "$HOLDOUT_SEC" || exit 1
elif [ "$LANE" = reco24k ]; then
    # Amendment 2 / S11: 3x training data, otherwise identical to the registered recogniser.
    train R24k-train outputs/lora_stepreco_24k "${TRAIN[@]}" --max_windows 24000 --seed 42 || exit 1
    for split in val test; do
        infer "R24k-$split" "$OUT/stepreco_24k_${split}_preds.json" --split "$split" "${RECO[@]}" \
            --lora_adapter outputs/lora_stepreco_24k || exit 1
    done
else
    for s in 1 2; do
        train "S$s-train" "outputs/lora_stepreco_s$s" "${TRAIN[@]}" --seed "$s" || exit 1
        for split in val test; do
            infer "S$s-$split" "$OUT/stepreco_s${s}_${split}_preds.json" --split "$split" "${RECO[@]}" \
                --lora_adapter "outputs/lora_stepreco_s$s" || exit 1
        done
    done
    train H-train outputs/lora_stepreco_holdout "${TRAIN[@]}" --recipe_holdout "$HOLDOUT" || exit 1
    for split in val test; do
        infer "H-$split" "$OUT/stepreco_holdout_${split}_preds.json" --split "$split" "${RECO[@]}" \
            --lora_adapter outputs/lora_stepreco_holdout --recipe_holdout "$HOLDOUT" \
            --step_seconds "$HOLDOUT_SEC" || exit 1
    done
fi
say "=== confirmation lane finished ==="
