#!/usr/bin/env bash
# Bounded, read-only A preflight. The shared environment and model cache are never edited.
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME PYTHONINSPECT PYTHONUSERBASE VIRTUAL_ENV
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

lock=/root/autodl-tmp/moe-research-gpu.lock
expected_lock_inode=2304:25841682495
expected_gpu=GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e
python=/root/autodl-tmp/moe-a-runtime-20260930/venv/bin/python
pkg=/root/autodl-tmp/moe-a-pkg-20260930/candidate_ltr_r02_env
verifier=/root/autodl-tmp/moe-a-shared-assets-launch-20260930-v2/verify_shared_model_20260930.py
receipt=/root/autodl-tmp/moe-a-shared-model-preflight-20260930.json
status_file=/root/autodl-tmp/moe-a-shared-model-preflight-status-20260930.json
expected_preflight_sha=4b834f2e3e96152f31e2c88524d5420401887e3f3b56e2c0df4ef826b3537cf8
expected_verifier_sha=d78c6726a8d010f3a02d50c79220084520c186f9b05f4b4a10c1598ac5fe9a90
expected_revision=6d84c48581ece794365f2b8e9cfb043c68ade9c5
expected_cache=/root/autodl-tmp/c-research-20260930/hf-cache

[[ -f "$lock" && ! -L "$lock" ]] || { echo 'shared lock missing' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'shared lock identity changed' >&2; exit 75;
}
[[ ! -e "$receipt" && ! -L "$receipt" && ! -e "$status_file" && ! -L "$status_file" ]] || {
  echo 'A model preflight already attempted' >&2; exit 75;
}
exec 9<>"$lock"
flock -n 9 || { echo 'GPU_LOCK_BUSY: defer A model preflight' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "/proc/$$/fd/9")" = "$expected_lock_inode" && \
   "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'held lock or path differs after acquisition' >&2; exit 75;
}
[[ ! -e "$receipt" && ! -L "$receipt" && ! -e "$status_file" && ! -L "$status_file" ]] || {
  echo 'A model preflight already attempted' >&2; exit 75;
}
stage=SOURCE_SHA256
write_status() {
  local state=$1 code=$2 status_tmp_dir
  status_tmp_dir=$(mktemp -d "${status_file}.tmp.XXXXXXXX") || return 1
  if ! printf '{"status":"%s","exit_code":%s,"last_stage":"%s"}\n' \
    "$state" "$code" "$stage" >"$status_tmp_dir/status.json"; then
    echo "status receipt could not be written: $status_tmp_dir" >&2
    return 1
  fi
  if ! mv -T -- "$status_tmp_dir/status.json" "$status_file"; then
    echo "status receipt could not be published: $status_tmp_dir" >&2
    return 1
  fi
  rmdir -- "$status_tmp_dir"
}
finish() {
  local code=$?
  trap - EXIT
  if (( code == 0 )); then
    write_status VERIFIED_READ_ONLY 0 || { echo 'final status receipt write failed' >&2; exit 75; }
  else
    write_status INCOMPLETE "$code" || { echo 'failure status receipt write failed' >&2; exit 75; }
  fi
  exit "$code"
}
trap finish EXIT
write_status STARTED 0
check_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "pinned source missing or is a symlink: $path" >&2; return 75;
  }
  actual=$(sha256sum -- "$path") || return 75
  [[ "${actual%% *}" = "$expected" ]] || {
    echo "pinned source SHA-256 changed: $path" >&2; return 75;
  }
}
[[ -x "$python" ]] || { echo 'pinned Python missing' >&2; exit 75; }
check_sha "$pkg/pkg/preflight.py" "$expected_preflight_sha"
check_sha "$verifier" "$expected_verifier_sha"
stage=GPU_IDENTITY
if ! gpu_uuid=$(timeout --signal=TERM --kill-after=5s 20s \
  nvidia-smi --query-gpu=uuid --format=csv,noheader); then
  echo 'GPU identity query failed or timed out' >&2; exit 75
fi
gpu_uuid=${gpu_uuid//$'\r'/}
gpu_uuid=${gpu_uuid// /}
[[ "$gpu_uuid" = "$expected_gpu" ]] || {
  echo 'authorized GPU identity changed' >&2; exit 75;
}
if ! gpu_processes=$(timeout --signal=TERM --kill-after=5s 20s \
  nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
  echo 'GPU process query failed or timed out' >&2; exit 75
fi
[[ -z "$gpu_processes" ]] || { echo 'GPU compute process still active' >&2; exit 75; }
stage=PRIVATE_PYTHON_IDENTITY
timeout --signal=TERM --kill-after=5s 60s "$python" - <<'PY'
import json
from pathlib import Path
import sys

import torch
import transformers
import vllm

private = Path("/root/autodl-tmp/moe-a-runtime-20260930/venv").resolve()
source = Path("/root/autodl-tmp/c-vllm-v026-venv").resolve()
status_file = Path("/root/autodl-tmp/moe-a-runtime-clone-status-20260930.json")
if not status_file.is_file() or status_file.is_symlink():
    raise SystemExit("A private clone status receipt missing")
status = json.loads(status_file.read_text())
if (status.get("status") != "CLONED_CPU_VERIFIED"
        or status.get("exit_code") != 0
        or status.get("a_venv") != str(private)
        or status.get("source_venv") != str(source)
        or status.get("lock_device_inode") != "2304:25841682495"):
    raise SystemExit("A private clone status receipt differs")
if (sys.flags.optimize != 0 or Path(sys.prefix).resolve() != private
        or Path(sys.executable).parent.resolve() != private / "bin"):
    raise SystemExit("A private Python prefix or optimization differs")
for name, module in (("vllm", vllm), ("torch", torch), ("transformers", transformers)):
    if not Path(module.__file__).resolve().is_relative_to(private):
        raise SystemExit(f"{name} does not resolve within A's private venv")
for path_text in sys.path:
    if path_text and Path(path_text).resolve().is_relative_to(source):
        raise SystemExit("A private Python path includes C runtime")
versions = (sys.version_info[:2], vllm.__version__, transformers.__version__,
            torch.__version__, torch.version.cuda)
if versions != ((3, 12), "0.26.0", "5.15.1", "2.11.0+cu130", "13.0"):
    raise SystemExit(f"A private runtime versions differ: {versions!r}")
PY
stage=VLLM_SOURCE_HASHES
timeout --signal=TERM --kill-after=5s 60s "$python" "$pkg/pkg/preflight.py"
stage=MODEL_SHA256
receipt_tmp_dir=$(mktemp -d "${receipt}.tmp.XXXXXXXX")
if timeout --signal=TERM --kill-after=5s 300s "$python" "$verifier" | \
  tee "$receipt_tmp_dir/receipt.json"; then
  verifier_rc=0
else
  verifier_rc=$?
fi
[[ -f "$receipt_tmp_dir/receipt.json" && ! -L "$receipt_tmp_dir/receipt.json" ]] || {
  echo "model receipt was not captured: $receipt_tmp_dir" >&2; exit 75;
}
[[ ! -e "$receipt" && ! -L "$receipt" ]] || {
  echo 'model receipt path appeared during preflight' >&2; exit 75;
}
ln -T -- "$receipt_tmp_dir/receipt.json" "$receipt" || {
  echo "model receipt could not be published; raw output remains in $receipt_tmp_dir" >&2
  exit 75
}
rm -- "$receipt_tmp_dir/receipt.json"
rmdir -- "$receipt_tmp_dir"
(( verifier_rc == 0 )) || { echo 'model verifier failed' >&2; exit "$verifier_rc"; }
stage=RECEIPT_VALIDATION
timeout --signal=TERM --kill-after=5s 60s "$python" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
if (not isinstance(data, dict)
        or data.get("status") != "VERIFIED_READ_ONLY"
        or data.get("revision") != sys.argv[2]
        or data.get("cache") != sys.argv[3]):
    raise SystemExit("model receipt status, revision, or cache differs")
' "$receipt" "$expected_revision" "$expected_cache"
stage=COMPLETE
