#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo 'usage: run.sh {ltr_t30_q1} ABSOLUTE_OUTPUT_DIR' >&2
  exit 64
fi
arm=$1
output_dir=$2
[[ "$output_dir" = /* ]] || { echo 'output directory must be absolute' >&2; exit 64; }
case "$arm" in
  ltr_t30_q1) ;;
  *) echo 'unsupported G64 admission diagnostic arm' >&2; exit 64 ;;
esac

: "${GPERF_AUTHORIZED_GPU_UUID:?explicit GPU UUID required}"
: "${GPERF_PYTHON:?pinned runtime Python required}"
: "${GPERF_HF_CACHE_DIR:?existing offline model cache required}"
: "${GPERF_LOCK_PATH:?shared GPU lock path required}"
: "${GPERF_MAX_WALL_SECONDS:?approved per-cell wall limit required}"
: "${GPERF_APPROVED_HOST_BYTES:?approved process-tree host memory limit required}"
: "${GPERF_CGROUP_MEMORY_MAX_FILE:?process-tree cgroup memory limit file required}"
: "${GPERF_EXPECTED_MANIFEST_SHA256:?root-frozen candidate manifest SHA-256 required}"
[[ "$GPERF_LOCK_PATH" = /* ]] || { echo 'lock path must be absolute' >&2; exit 64; }
[[ -f "$GPERF_LOCK_PATH" && ! -L "$GPERF_LOCK_PATH" ]] || { echo 'shared lock must already exist as a regular file' >&2; exit 64; }
[[ "$GPERF_PYTHON" = /* && -x "$GPERF_PYTHON" ]] || { echo 'pinned Python must be an absolute executable path' >&2; exit 64; }
[[ -d "$GPERF_HF_CACHE_DIR" ]] || { echo 'offline model cache directory missing' >&2; exit 64; }
[[ "$GPERF_EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid expected manifest SHA-256' >&2; exit 64; }
[[ "$GPERF_MAX_WALL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'wall limit must be positive seconds' >&2; exit 64; }
(( GPERF_MAX_WALL_SECONDS > 6 )) || { echo 'wall limit too short for bounded shutdown' >&2; exit 64; }
[[ "$GPERF_APPROVED_HOST_BYTES" =~ ^[1-9][0-9]*$ ]] || { echo 'host limit must be positive bytes' >&2; exit 64; }
[[ -r /proc/self/cgroup && -d /sys/fs/cgroup ]] || { echo 'current process cgroup unavailable' >&2; exit 64; }
[[ "$(stat -fc %T /sys/fs/cgroup)" = cgroup2fs ]] || { echo 'requires unified cgroup v2 memory limit' >&2; exit 64; }
own_cgroup=$(awk -F: '$1 == "0" && $2 == "" {print $3}' /proc/self/cgroup)
[[ "$own_cgroup" = /* && "$own_cgroup" != *..* ]] || { echo 'cannot identify current process cgroup' >&2; exit 64; }
own_memory_max="/sys/fs/cgroup${own_cgroup%/}/memory.max"
[[ -r "$GPERF_CGROUP_MEMORY_MAX_FILE" ]] || { echo 'cgroup memory limit unreadable' >&2; exit 64; }
[[ "$(realpath -e -- "$GPERF_CGROUP_MEMORY_MAX_FILE")" = "$(realpath -e -- "$own_memory_max")" ]] || {
  echo 'memory limit file does not belong to current process cgroup' >&2; exit 64;
}
read -r actual_host_limit < "$GPERF_CGROUP_MEMORY_MAX_FILE"
[[ "$actual_host_limit" =~ ^[1-9][0-9]*$ ]] || { echo 'cgroup memory is unbounded' >&2; exit 64; }
(( actual_host_limit <= GPERF_APPROVED_HOST_BYTES )) || { echo 'cgroup memory exceeds approved limit' >&2; exit 64; }

export HF_HOME="$GPERF_HF_CACHE_DIR" HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPERF_AUTHORIZED_GPU_UUID"
started_s=$(date +%s)
cd -- "$(dirname -- "$0")"
package_root=$(dirname "$(pwd -P)")
resolved_output=$(realpath -m -- "$output_dir")
[[ "$resolved_output" != "$package_root" && "$resolved_output" != "$package_root"/* ]] || {
  echo 'output directory must be outside candidate package' >&2; exit 64;
}
if [[ -e /proc/self/fd/9 ]]; then
  [[ "$(stat -Lc '%d:%i' /proc/self/fd/9)" = "$(stat -Lc '%d:%i' "$GPERF_LOCK_PATH")" ]] || {
    echo 'inherited lock descriptor points to another file' >&2; exit 75;
  }
else
  exec 9<>"$GPERF_LOCK_PATH"
fi
flock -n 9 || { echo 'ABORT_GPU_GROUP_LOCK_BUSY' >&2; exit 75; }
run_bounded() {
  local remaining_s
  remaining_s=$(( GPERF_MAX_WALL_SECONDS - ($(date +%s) - started_s) ))
  (( remaining_s > 5 )) || { echo 'wall budget exhausted before stage' >&2; return 75; }
  timeout --signal=TERM --kill-after=5s "$((remaining_s - 5))s" "$@"
}
actual_manifest_sha=$(sha256sum ../manifest.json)
[[ "${actual_manifest_sha%% *}" = "$GPERF_EXPECTED_MANIFEST_SHA256" ]] || {
  echo 'candidate manifest identity differs from root-frozen SHA-256' >&2; exit 75;
}
gpu_uuids=$(run_bounded nvidia-smi --query-gpu=uuid --format=csv,noheader)
matched_gpu=0
while IFS= read -r uuid; do
  uuid=${uuid// /}
  uuid=${uuid//$'\r'/}
  if [[ "$uuid" = "$GPERF_AUTHORIZED_GPU_UUID" ]]; then matched_gpu=$((matched_gpu + 1)); fi
done <<< "$gpu_uuids"
(( matched_gpu == 1 )) || { echo 'authorized GPU UUID absent or ambiguous' >&2; exit 75; }
mkdir "../launch-once-${arm}" || { echo 'G64 admission diagnostic identity already consumed' >&2; exit 75; }
run_bounded "$GPERF_PYTHON" ../verify_package.py
run_bounded "$GPERF_PYTHON" -c 'import sys; sys.exit(0 if sys.flags.optimize == 0 else 1)' || {
  echo 'pinned Python optimization would disable source assertions' >&2; exit 75;
}
run_bounded "$GPERF_PYTHON" preflight.py

run_bounded "$GPERF_PYTHON" run_ltr_style.py --inputs inputs --warmup-inputs warmups \
  --ltr-threshold 30 --ltr-quantum 1 --measurement-mode admission_diagnostic \
  --output-dir "$output_dir"
