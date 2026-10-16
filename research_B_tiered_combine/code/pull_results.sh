#!/usr/bin/env bash
# Pull experiment artifacts; SSH asks for credentials, never stores a password.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOST=${EXPERIMENT_HOST:-root@connect.westb.seetacloud.com}
PORT=${EXPERIMENT_PORT:-50328}
REMOTE=${EXPERIMENT_ROOT:-/root/autodl-tmp/research_B_tiered_combine}
SOCKET_DIR=$(mktemp -d /tmp/tiered-sync.XXXXXX)
SOCKET="$SOCKET_DIR/ssh"
cleanup() {
  ssh -S "$SOCKET" -O exit -p "$PORT" "$HOST" >/dev/null 2>&1 || true
  rm -rf "$SOCKET_DIR"
}
trap cleanup EXIT
ssh -M -S "$SOCKET" -o ControlPersist=60 -o ConnectTimeout=20 -fnN -p "$PORT" "$HOST"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ROOT/runs" "$ROOT/runs/.sync_history/$STAMP"
rsync -rt --checksum --partial --backup \
  --backup-dir="$ROOT/runs/.sync_history/$STAMP" \
  -e "ssh -S $SOCKET -p $PORT" \
  "$HOST:$REMOTE/runs/" "$ROOT/runs/"
# An empty itemized dry run means all current remote files match local bytes.
CHANGES=$(rsync -rcni -e "ssh -S $SOCKET -p $PORT" "$HOST:$REMOTE/runs/" "$ROOT/runs/")
if [ -n "$CHANGES" ]; then
  printf '%s\n' "$CHANGES"
  echo 'Remote files changed during transfer; run again after the experiment finishes.' >&2
  exit 2
fi
printf '%s remote results match local checksums (%s:%s/runs/)\n' "$STAMP" "$HOST" "$REMOTE" | tee -a "$ROOT/runs/sync.log"
