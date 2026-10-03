#!/usr/bin/env bash
# One bounded download of the already frozen OLMoE revision. No GPU model load.
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME PYTHONINSPECT
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

root=/root/autodl-tmp/moe-a-20260930
venv=/root/autodl-tmp/c-vllm-v026-venv
pkg=/root/autodl-tmp/moe-a-pkg-20260930/candidate_ltr_r02_env
lock=/root/autodl-tmp/moe-research-gpu.lock
expected_lock_inode=2304:25841682495
revision=6d84c48581ece794365f2b8e9cfb043c68ade9c5

[[ -f "$lock" && ! -L "$lock" ]] || { echo 'common lock missing' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'common lock inode changed' >&2; exit 75;
}
exec 9<>"$lock"
flock -n 9 || { echo 'GPU_LOCK_BUSY: defer model download' >&2; exit 75; }
[[ "$(stat -Lc '%d:%i' "/proc/$$/fd/9")" = "$expected_lock_inode" && \
   "$(stat -Lc '%d:%i' "$lock")" = "$expected_lock_inode" ]] || {
  echo 'common lock fd or path inode changed after acquisition' >&2; exit 75;
}
[[ -d "$venv" && ! -L "$venv" && -x "$venv/bin/python" && -f "$pkg/pkg/preflight.py" ]] || {
  echo 'pinned environment or package missing' >&2; exit 75;
}
[[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '\r ')" = GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e ]] || {
  echo 'authorized GPU identity changed' >&2; exit 75;
}
if ! gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
  echo 'GPU process query failed: defer download' >&2; exit 75
fi
[[ -z "$gpu_processes" ]] || {
  echo 'GPU has a compute process: defer download' >&2; exit 75;
}
[[ "$(cat /sys/fs/cgroup/memory.max)" = 96636764160 ]] || {
  echo '90 GiB host cgroup changed' >&2; exit 75;
}
available=$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
(( available >= 20000000000 )) || { echo 'data disk has under 20 GB free' >&2; exit 75; }
timeout --signal=TERM --kill-after=5s 60s "$venv/bin/python" "$pkg/pkg/preflight.py"
timeout --signal=TERM --kill-after=5s 60s "$venv/bin/python" - <<'PY'
import sys, torch, vllm
from importlib.metadata import version
from pathlib import Path
venv = Path("/root/autodl-tmp/c-vllm-v026-venv").resolve()
if sys.flags.optimize != 0 or Path(sys.prefix).resolve() != venv:
    raise SystemExit("shared Python prefix or optimization differs")
if venv not in Path(vllm.__file__).resolve().parents:
    raise SystemExit("vLLM imported from outside the shared venv")
if (sys.version_info[:2], vllm.__version__, version("transformers"),
        torch.__version__, torch.version.cuda) != (
        (3, 12), "0.26.0", "5.15.1", "2.11.0+cu130", "13.0"):
    raise SystemExit("shared runtime versions differ from historical G/H")
print("Shared runtime versions match the historical G/H environment; no GPU initialized")
PY
[[ ! -e "$root/model-download-started" ]] || { echo 'one-shot model download already attempted' >&2; exit 75; }
[[ ! -e "$root" && ! -L "$root" ]] || { echo 'A model directory already exists' >&2; exit 75; }
mkdir -m 700 "$root"
mkdir -m 700 "$root/tmp"
touch "$root/model-download-started"
export HF_HOME="$root/hf" HF_HUB_DOWNLOAD_TIMEOUT=60 TMPDIR="$root/tmp"
date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ' | tee "$root/model-receipt.txt"
timeout --signal=TERM --kill-after=10s 2400s "$venv/bin/python" - <<PY >"$root/model-download.log" 2>&1 || {
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="allenai/OLMoE-1B-7B-0924",
    revision="$revision",
    cache_dir="$root/hf/hub",
    max_workers=2,
)
print(path)
PY
  status=$?
  tail -40 "$root/model-download.log" >&2
  echo "model_download_exit=$status" >>"$root/model-receipt.txt"
  exit "$status"
}
snapshot="$root/hf/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/$revision"
[[ -f "$snapshot/config.json" && -f "$snapshot/model.safetensors.index.json" ]] || {
  echo 'fixed model snapshot is incomplete' >&2; exit 75;
}
"$venv/bin/python" - "$snapshot" <<'PY' | tee -a "$root/model-receipt.txt"
import hashlib, json, pathlib, sys
snapshot = pathlib.Path(sys.argv[1])
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
expected = {
    "model-00001-of-00003.safetensors": (4997744872, "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"),
    "model-00002-of-00003.safetensors": (4997235176, "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"),
    "model-00003-of-00003.safetensors": (3843741912, "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"),
}
if set(shards) != set(expected):
    raise SystemExit("fixed model shard names differ from historical preflight")
print("snapshot=" + str(snapshot))
for name in shards:
    path = snapshot / name
    size, expected_sha = expected[name]
    if not path.is_file() or path.stat().st_size != size:
        raise SystemExit(f"fixed model shard missing or size differs: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise SystemExit(f"fixed model shard SHA-256 differs: {name}")
    print(f"{name}={size} sha256={expected_sha}")
PY
date -u '+completed_utc=%Y-%m-%dT%H:%M:%SZ' | tee -a "$root/model-receipt.txt"
