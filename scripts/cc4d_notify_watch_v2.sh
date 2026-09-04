#!/usr/bin/env bash
# Stage notifier for the graph-ablation chain.
#
# Deliberately a SEPARATE process that tails the chain log, rather than calls
# added inside the chain: editing a running bash script is unsafe (bash reads it
# incrementally by byte offset), and the chain is mid-run. This can also be
# restarted or killed without touching the experiment.
#
# notify-send does NOT work on this host -- headless SSH, no notification daemon
# on the session bus (org.freedesktop.Notifications is unclaimed). It is still
# attempted last, harmlessly, in case a desktop session appears later. The
# channels that actually deliver here are tmux and the marker file.
#
#   nohup bash scripts/cc4d_notify_watch.sh &
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=${CHAIN_LOG:-logs/cc4d_graph_ablation_chain.log}
MARK=${STAGE_MARK:-outputs/decision_v2/STAGE_LOG.txt}
touch "$MARK"

notify() {
    local msg="$1"
    # De-duplicate. `tail -F` follows by NAME, so anything that replaces the log
    # file's inode -- an editor, sed -i, log rotation -- makes it reopen and
    # replay the entire history stamped with the CURRENT time. That once made
    # already-finished stages look as though they had just completed. Never
    # report the same message text twice.
    if [ -f "$MARK" ] && grep -qF -- "$msg" "$MARK"; then return; fi
    # 1. Durable: survives detach, readable on return.
    echo "[$(date '+%F %T')] $msg" >> "$MARK"
    # 2. tmux: visible in the status line of every session, attached or not.
    if command -v tmux >/dev/null && tmux list-sessions >/dev/null 2>&1; then
        tmux list-sessions -F '#{session_name}' 2>/dev/null | while read -r s; do
            tmux display-message -t "$s" "GraphWalk: $msg" 2>/dev/null
        done
    fi
    # 3. Best effort, expected to fail on this host.
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
        notify-send "GraphWalk chain" "$msg" 2>/dev/null || true
}

notify "notifier armed $(date +%H:%M:%S)"

tail -n0 -F "$LOG" 2>/dev/null | stdbuf -oL tr '\r' '\n' | while IFS= read -r line; do
    case "$line" in
        *"CUT LINE"*)        notify "${line#*] }" ;;
        *"eval done in"*)    ;;   # per-run timing noise, not a stage marker
        *" done"*)           notify "${line#*] }" ;;
        *FAILED*|*aborting*) notify "FAILURE -- ${line#*] }" ;;
        *"chain finished"*)  notify "ALL STAGES COMPLETE" ;;
        *Traceback*|*"out of memory"*|*Killed*)
                             notify "ERROR -- $line" ;;
    esac
done
