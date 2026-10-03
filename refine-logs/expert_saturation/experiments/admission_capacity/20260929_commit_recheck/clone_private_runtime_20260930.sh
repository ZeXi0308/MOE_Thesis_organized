#!/usr/bin/env bash
# One nonblocking, reflink-only copy of the pinned C runtime into A's private path.
# The source venv is only read; this script never initializes a GPU model.
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME PYTHONINSPECT PYTHONUSERBASE VIRTUAL_ENV
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

lock=/root/autodl-tmp/moe-research-gpu.lock
expected_lock_inode=2304:25841682495
expected_gpu=GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e
source_venv=/root/autodl-tmp/c-vllm-v026-venv
a_root=/root/autodl-tmp/moe-a-runtime-20260930
a_venv="$a_root/venv"
pkg=/root/autodl-tmp/moe-a-pkg-20260930/candidate_ltr_r02_env
expected_manifest=f58340cd5228ea7339f0d4081176aab265583cb824a26026d19b8f7ae93cc8a7
status_file=/root/autodl-tmp/moe-a-runtime-clone-status-20260930.json
stage=PRE_LOCK

[[ -f "$lock" && ! -L "$lock" ]] || { echo 'shared lock missing or invalid' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'shared lock identity changed' >&2; exit 75;
}
[[ ! -e "$a_root" && ! -L "$a_root" && ! -e "$status_file" && ! -L "$status_file" ]] || {
  echo 'one-shot A runtime clone path or receipt already exists' >&2; exit 75;
}
exec 9<>"$lock"
flock -n 9 || { echo 'GPU_LOCK_BUSY: defer A runtime clone' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "/proc/$$/fd/9")" = "$expected_lock_inode" && \
   "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'held lock or path differs from frozen inode' >&2; exit 75;
}
[[ ! -e "$a_root" && ! -L "$a_root" && ! -e "$status_file" && ! -L "$status_file" ]] || {
  echo 'one-shot A runtime clone path or receipt appeared under lock' >&2; exit 75;
}

write_status() {
  local state=$1 code=$2 stamp tmp
  stamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  tmp="$status_file.tmp.$$"
  printf '{"status":"%s","exit_code":%s,"last_stage":"%s","utc":"%s","source_venv":"%s","a_venv":"%s","lock_device_inode":"%s"}\n' \
    "$state" "$code" "$stage" "$stamp" "$source_venv" "$a_venv" "$expected_lock_inode" >"$tmp"
  mv -- "$tmp" "$status_file"
}
finish() {
  local code=$?
  trap - EXIT
  if (( code == 0 )); then
    write_status CLONED_CPU_VERIFIED 0
  else
    write_status INCOMPLETE "$code"
  fi
}
trap 'exit 130' INT
trap 'exit 143' TERM
write_status STARTED 0
trap finish EXIT

stage=LIVE_RESOURCE_GATE
if ! gpu_uuids=$(timeout --signal=TERM --kill-after=5s 15s \
    nvidia-smi --query-gpu=uuid --format=csv,noheader); then
  echo 'GPU UUID query failed or timed out' >&2; exit 75
fi
[[ "$(printf '%s' "$gpu_uuids" | tr -d '\r ')" = "$expected_gpu" ]] || {
  echo 'authorized GPU identity changed' >&2; exit 75;
}
if ! gpu_processes=$(timeout --signal=TERM --kill-after=5s 15s \
    nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
  echo 'GPU process query failed or timed out' >&2; exit 75
fi
[[ -z "$gpu_processes" ]] || { echo 'GPU compute process active' >&2; exit 75; }
export CUDA_VISIBLE_DEVICES="$expected_gpu"
[[ "$(cat /sys/fs/cgroup/memory.max)" = 96636764160 ]] || {
  echo '90 GiB host cgroup changed' >&2; exit 75;
}

stage=SOURCE_AND_PACKAGE_GATE
[[ -d "$source_venv" && ! -L "$source_venv" && -x "$source_venv/bin/python" ]] || {
  echo 'completed shared runtime is missing or symlinked' >&2; exit 75;
}
[[ -d "$pkg" && ! -L "$pkg" && -f "$pkg/manifest.json" && -f "$pkg/verify_package.py" ]] || {
  echo 'frozen A package is missing' >&2; exit 75;
}
manifest_sha=$(sha256sum "$pkg/manifest.json")
[[ "${manifest_sha%% *}" = "$expected_manifest" ]] || {
  echo 'A package manifest differs from frozen identity' >&2; exit 75;
}
available=$(timeout --signal=TERM --kill-after=5s 15s \
  df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
[[ "$available" =~ ^[0-9]+$ ]] && (( available >= 4000000000 )) || {
  echo 'data disk has under 4 GB free or free-space query failed' >&2; exit 75;
}

stage=CREATE_A_ROOT
mkdir -m 700 "$a_root"
mkdir -m 700 "$a_root/tmp" "$a_root/cache" "$a_root/hf"
export TMPDIR="$a_root/tmp" XDG_CACHE_HOME="$a_root/cache" HF_HOME="$a_root/hf"
export TORCHINDUCTOR_CACHE_DIR="$a_root/cache/torchinductor"
timeout --signal=TERM --kill-after=5s 60s "$source_venv/bin/python" "$pkg/verify_package.py" \
  >"$a_root/package-check.log" 2>&1

verify_env() {
  local python_bin=$1 expected_prefix=$2 role=$3
  timeout --signal=TERM --kill-after=5s 120s "$python_bin" - \
    "$expected_prefix" "$source_venv" "$pkg/pkg/runtime_source_hashes.json" "$role" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import torch
import transformers
import vllm

expected = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
hash_file = Path(sys.argv[3])
role = sys.argv[4]
if sys.flags.optimize != 0 or Path(sys.prefix).resolve() != expected:
    raise SystemExit("Python prefix or optimization differs")
if Path(sys.executable).parent.resolve() != expected / "bin":
    raise SystemExit("Python executable is outside the expected venv")
module_paths = {}
for name, module in (("torch", torch), ("transformers", transformers), ("vllm", vllm)):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(expected):
        raise SystemExit(f"{name} imported from outside the expected venv: {path}")
    module_paths[name] = str(path)
if role == "CLONE":
    for path_text in sys.path:
        if path_text and Path(path_text).resolve().is_relative_to(source):
            raise SystemExit(f"clone sys.path still includes C runtime: {path_text}")
versions = (sys.version_info[:2], vllm.__version__, transformers.__version__,
            torch.__version__, torch.version.cuda)
required = ((3, 12), "0.26.0", "5.15.1", "2.11.0+cu130", "13.0")
if versions != required:
    raise SystemExit(f"runtime versions differ: {versions!r}")
hashes = json.loads(hash_file.read_text())
if len(hashes) != 8:
    raise SystemExit("expected eight frozen vLLM source hashes")
vllm_root = Path(vllm.__file__).resolve().parent
for relative, sha in hashes.items():
    actual = hashlib.sha256((vllm_root / relative).read_bytes()).hexdigest()
    if actual != sha:
        raise SystemExit(f"vLLM source SHA-256 differs: {relative}")
print(json.dumps({"status": "SOURCE_MATCH", "role": role,
                  "python_prefix": str(expected), "versions": list(versions),
                  "module_paths": module_paths, "source_hashes": hashes}, sort_keys=True))
PY
}

stage=VERIFY_C_SOURCE
verify_env "$source_venv/bin/python" "$source_venv" SOURCE \
  >"$a_root/source-check.json" 2>"$a_root/source-check.err"

stage=REFLINK_CLONE
timeout --signal=TERM --kill-after=10s 1800s \
  cp -a --reflink=always -- "$source_venv" "$a_venv" \
  >"$a_root/clone.log" 2>&1
[[ -d "$a_venv" && ! -L "$a_venv" && -x "$a_venv/bin/python" ]] || {
  echo 'A venv clone missing after cp' >&2; exit 75;
}

stage=VERIFY_A_CLONE
verify_env "$a_venv/bin/python" "$a_venv" CLONE \
  >"$a_root/clone-check.json" 2>"$a_root/clone-check.err"
if ! gpu_processes=$(timeout --signal=TERM --kill-after=5s 15s \
    nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
  echo 'GPU process query failed or timed out after clone' >&2; exit 75
fi
[[ -z "$gpu_processes" ]] || { echo 'GPU compute process active after clone' >&2; exit 75; }
stage=COMPLETE
