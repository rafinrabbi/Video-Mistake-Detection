#!/usr/bin/env bash
# Relaunch every confirmation lane that is neither running nor finished.
# Safe to run repeatedly; each lane resumes from its own checkpoints.
# Installed as an @reboot crontab entry; stop with outputs/confirm_v1/STOP_CHAIN.
#
#   bash scripts/cc4d_resume_lanes.sh
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f outputs/confirm_v1/STOP_CHAIN ] && exit 0
mkdir -p logs
for lane in infer train reco24k; do
    log=logs/cc4d_confirm_${lane}.log
    [ -f "$log" ] || continue                       # lane never started: not ours to start
    last=$(tr '\r' '\n' < "$log" | grep -aE "^\[20[0-9-]+ [0-9:]+\] \[$lane\] " | tail -1)
    case "$last" in *"=== confirmation lane finished ==="*) continue ;; esac
    pgrep -f "cc4d_confirm_chain.sh $lane\$" > /dev/null && continue
    echo "[$(date '+%F %T')] [resume] relaunching lane $lane" >> logs/cc4d_resume_lanes.log
    setsid nohup bash scripts/cc4d_confirm_chain.sh "$lane" > /dev/null 2>&1 < /dev/null &
done
