#!/bin/bash
# Run the eight length-dispersion cells, waiting for the GPU rather than competing.
#
# Another session on this host runs its own campaigns on the same GPU. Racing it
# would corrupt both sets of timings, and killing its processes is not an option,
# so each cell here waits for the device to go idle, then re-checks immediately
# before loading the model. If the device is taken again in that window the cell
# simply goes back to waiting; nothing is forced and no foreign process is
# signalled.
#
# A cell whose output directory already exists is skipped, so this script can be
# restarted at any time without overwriting a completed measurement.
set -u
cd /root/het-r01/pkg
V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
R=/root/het-r01/results
MAX_WAIT_S=${MAX_WAIT_S:-5400}

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]'
}

wait_for_gpu() {  # label
  local waited=0
  while [ -n "$(gpu_pids)" ]; do
    if [ "$waited" -ge "$MAX_WAIT_S" ]; then
      echo "GIVE_UP $1: GPU still busy after ${waited}s (PIDs $(gpu_pids))"
      return 1
    fi
    [ $((waited % 300)) -eq 0 ] && echo "    $(date -u +%H:%M:%S) waiting for GPU, held by $(gpu_pids)"
    sleep 30
    waited=$((waited + 30))
  done
  return 0
}

run() {  # label arm policy
  local label=$1 arm=$2 policy=$3
  if [ -d "$R/$label" ]; then
    echo "SKIP $label (already present)"
    return 0
  fi
  wait_for_gpu "$label" || return 1
  # Re-check right before the load; a foreign process may have started while
  # we were sleeping.
  if [ -n "$(gpu_pids)" ]; then
    echo "RETRY $label: taken during handoff"
    run "$label" "$arm" "$policy"
    return $?
  fi
  echo "=== $(date -u +%H:%M:%S) START $label ($arm / $policy) ==="
  timeout 700 "$V" -u run_probe.py --domain pair --arm "$arm" \
      --reservation-policy full --completion-policy "$policy" --cap 42 \
      --gpu-memory-utilization 0.9 --output-dir "$R/$label" > "$R/$label.log" 2>&1
  local rc=$?
  echo "=== $(date -u +%H:%M:%S) DONE $label exit=$rc ==="
  "$V" - "$R/$label/status.json" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print("   ", d.get("status"), "completed", d.get("requests_completed"),
      "preemptions", d.get("actual_preemption_count"), "err", d.get("error"))
PY
}

run block0-hom-native   homogeneous   native
run block0-hom-rotate   homogeneous   rotate
run block0-het-native   heterogeneous native
run block0-het-rotate   heterogeneous rotate
run block1-het-rotate   heterogeneous rotate
run block1-het-native   heterogeneous native
run block1-hom-rotate   homogeneous   rotate
run block1-hom-native   homogeneous   native

echo "ALL DONE $(date -u +%H:%M:%S)"
