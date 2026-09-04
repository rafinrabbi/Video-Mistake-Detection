#!/usr/bin/env bash
# Bridge the chain's real output back to the visible log path.
#
# Why this is needed: the chain opened its log with `exec >> LOG` and the trainer
# inherited that descriptor. Editing that file in place with `sed -i` replaced
# the INODE, so both processes went on writing into a now-deleted file while the
# path they are supposed to write to sat frozen. Everything watching the path --
# the notifier, the monitor, a human running tail -f -- went silent while the run
# continued perfectly well. Silence that looks identical to progress is the worst
# possible failure mode for a multi-day job.
#
# The running processes' descriptors cannot be repointed without attaching a
# debugger, but the live inode is still readable through /proc/<pid>/fd/1, so we
# copy new bytes from there back to the path. Starts at the deleted file's
# current EOF, so nothing already in the visible log is duplicated.
#
#   nohup bash scripts/cc4d_log_bridge.sh <chain_pid> &
set -uo pipefail
cd "$(dirname "$0")/.."
PID="${1:?usage: cc4d_log_bridge.sh <chain_pid>}"
LOG=logs/cc4d_graph_ablation_chain.log
SRC="/proc/$PID/fd/1"

[ -e "$SRC" ] || { echo "no such fd: $SRC" >&2; exit 1; }
echo "[$(date '+%F %T')] --- log bridge attached to pid $PID (previous output went to a deleted inode) ---" >> "$LOG"
exec tail -c 0 -f "$SRC" >> "$LOG"
