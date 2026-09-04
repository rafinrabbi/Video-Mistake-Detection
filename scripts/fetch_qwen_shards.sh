#!/usr/bin/env bash
# Resumable fetch of the missing Qwen2.5-VL-7B safetensors shards.
#
# Why this exists instead of `huggingface-cli download`: on this host the HF CDN
# caps a transfer at ~0.5 MB/s and drops long-lived connections, so both the Xet
# backend and plain snapshot_download stall at zero bytes on multi-GB files.
#
# Why explicit byte ranges instead of `curl -C -`: the resolve URL 302s to a
# per-request signed CDN URL that does not reliably honour Range on resume. A
# resume that comes back 200 instead of 206 makes curl append a fresh full copy
# onto the existing prefix, producing a file of plausible size and wrong content
# (observed: shard 2 "completed" 3.1 GB in 37 s, sha256 mismatch).
#
# So: fetch fixed-size parts, each of which MUST return 206 with exactly the
# requested byte count or it is discarded and retried. Parts are independent, so
# a failure can never corrupt a neighbour, and an interrupted run resumes at
# part granularity. The concatenation is sha256-verified against the blob name
# before it is linked into the snapshot, so a bad result can never be used.
set -uo pipefail

REPO=Qwen/Qwen2.5-VL-7B-Instruct
CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
BLOBS="$CACHE/blobs"
PARTS="$CACHE/parts"
SNAP=$(ls -d "$CACHE"/snapshots/*/ | head -1)
TOKEN=$(cat "$HOME/.cache/huggingface/token")
PART_SIZE=$((32 * 1024 * 1024))   # 32 MiB: ~60 s at the observed rate
MAXTIME=600                        # per-part ceiling
MAX_RETRY=50                       # per part; the link is flaky, not dead

declare -A SHA=(
  [2]=a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024
  [4]=ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed
)

mkdir -p "$PARTS"
say() { echo "[$(date '+%F %T')] $*"; }

for n in 2 4; do
  name=$(printf 'model-0000%d-of-00005.safetensors' "$n")
  sha=${SHA[$n]}
  blob="$BLOBS/$sha"
  url="https://huggingface.co/$REPO/resolve/main/$name"

  if [ -e "$SNAP/$name" ] && [ -s "$blob" ]; then
    say "$name already present — skipping"; continue
  fi

  total=$(curl -sIL -H "Authorization: Bearer $TOKEN" "$url" \
          | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  [ -z "$total" ] && { say "FATAL: no content-length for $name"; exit 1; }
  nparts=$(( (total + PART_SIZE - 1) / PART_SIZE ))
  say "$name: $total bytes in $nparts parts of $((PART_SIZE/1024/1024)) MiB"

  i=0
  while [ "$i" -lt "$nparts" ]; do
    lo=$(( i * PART_SIZE ))
    hi=$(( lo + PART_SIZE - 1 )); [ "$hi" -ge "$total" ] && hi=$(( total - 1 ))
    want=$(( hi - lo + 1 ))
    pf="$PARTS/$sha.$i"

    if [ "$(stat -c %s "$pf" 2>/dev/null || echo 0)" -eq "$want" ]; then
      i=$(( i + 1 )); continue
    fi

    ok=0
    for try in $(seq 1 "$MAX_RETRY"); do
      code=$(curl -sL --max-time "$MAXTIME" \
             -H "Authorization: Bearer $TOKEN" -H "Range: bytes=$lo-$hi" \
             -o "$pf" -w '%{http_code}' "$url" 2>/dev/null)
      got=$(stat -c %s "$pf" 2>/dev/null || echo 0)
      # 206 with the exact byte count is the ONLY acceptable outcome. A 200 means
      # the range was ignored and the body is the whole file — discard it.
      if [ "$code" = "206" ] && [ "$got" -eq "$want" ]; then ok=1; break; fi
      rm -f "$pf"
      sleep 3
    done
    [ "$ok" -eq 1 ] || { say "FATAL: part $i of $name failed after $MAX_RETRY tries"; exit 1; }

    if [ $(( i % 20 )) -eq 0 ] || [ "$i" -eq $(( nparts - 1 )) ]; then
      say "$name: part $((i+1))/$nparts ($(( (i+1) * 100 / nparts ))%)"
    fi
    i=$(( i + 1 ))
  done

  say "$name: assembling $nparts parts"
  : > "$blob.assembling"
  for j in $(seq 0 $(( nparts - 1 ))); do cat "$PARTS/$sha.$j" >> "$blob.assembling"; done

  got=$(sha256sum "$blob.assembling" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then
    say "FATAL: checksum mismatch for $name (got $got) — parts kept at $PARTS"
    exit 1
  fi
  mv "$blob.assembling" "$blob"
  ln -sfn "../../blobs/$sha" "$SNAP/$name"
  rm -f "$PARTS/$sha".*
  say "$name: verified and linked"
done

rmdir "$PARTS" 2>/dev/null
say "all shards present"
