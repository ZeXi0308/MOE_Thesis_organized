#!/usr/bin/env bash
# One bounded, host-specific provisioning attempt. No GPU model is initialized.
set -euo pipefail
umask 077

root=/root/autodl-tmp/moe-a-20260930
venv="$root/venv"
lock=/root/autodl-tmp/moe-research-gpu.lock
expected_lock_inode=2304:25841682495
expected_gpu=GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e

[[ -f "$lock" && ! -L "$lock" ]] || { echo 'common lock is absent or invalid' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'common lock inode changed' >&2; exit 75;
}
exec 9<>"$lock"
flock -n 9 || { echo 'GPU_LOCK_BUSY: defer provisioning' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "/proc/$$/fd/9")" = "$expected_lock_inode" && \
   "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'common lock fd or path inode changed after acquisition' >&2; exit 75;
}
[[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '\r ')" = "$expected_gpu" ]] || {
  echo 'authorized GPU identity changed' >&2; exit 75;
}
if ! gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
  echo 'GPU process query failed: defer provisioning' >&2; exit 75
fi
[[ -z "$gpu_processes" ]] || {
  echo 'GPU has a compute process: defer provisioning' >&2; exit 75;
}
[[ "$(cat /sys/fs/cgroup/memory.max)" = 96636764160 ]] || {
  echo '90 GiB host cgroup changed' >&2; exit 75;
}
[[ ! -e "$root" ]] || { echo 'one-shot environment path already exists' >&2; exit 75; }
mkdir -p "$root/tmp"
export TMPDIR="$root/tmp" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ' | tee "$root/receipt.txt"
/root/miniconda3/bin/python3 -m venv "$venv"
timeout --signal=TERM --kill-after=10s 1500s "$venv/bin/python" -m pip install \
  --no-cache-dir 'vllm==0.26.0' 'transformers==5.15.1' \
  >"$root/pip-install.log" 2>&1 || {
    status=$?
    tail -40 "$root/pip-install.log" >&2
    echo "pip_install_exit=$status" >>"$root/receipt.txt"
    exit "$status"
  }
"$venv/bin/python" - <<'PY' | tee -a "$root/receipt.txt"
import sys, torch, vllm
from importlib.metadata import version
print('python=' + sys.version.split()[0])
print('torch=' + torch.__version__)
print('torch_cuda=' + str(torch.version.cuda))
print('vllm=' + vllm.__version__)
print('transformers=' + version('transformers'))
PY
du -sh "$venv" | tee -a "$root/receipt.txt"
date -u '+completed_utc=%Y-%m-%dT%H:%M:%SZ' | tee -a "$root/receipt.txt"
