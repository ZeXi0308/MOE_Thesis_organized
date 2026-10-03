#!/bin/bash
# Run the structural-deficit sweep: four KV pool sizes x {native, rotate} x two blocks.
#
# Cell order within a block walks the deficits in the same sequence and flips the
# arm order between blocks, so the forward/reverse pair for each (deficit, arm)
# brackets any drift over the campaign. That pair is the only repetition here; it
# yields an observed repeat difference, not a significance test.
#
# Before each cell the GPU is checked for foreign compute processes. If one is
# present the script waits rather than competing, and never signals it. A cell
# whose output directory already exists is skipped, so the script is safe to
# restart after an interruption.
set -u
PKG=/root/psweep-r01/pkg
R=/root/psweep-r01/results
V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
MAX_WAIT_S=${MAX_WAIT_S:-3600}

cd "$PKG" || exit 1
mkdir -p "$R"

gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]'; }

wait_for_gpu() {
  local waited=0
  while [ -n "$(gpu_pids)" ]; do
    if [ "$waited" -ge "$MAX_WAIT_S" ]; then
      echo "GIVE_UP $1: GPU busy after ${waited}s (PIDs $(gpu_pids))"
      return 1
    fi
    [ $((waited % 300)) -eq 0 ] && echo "    $(date -u +%H:%M:%S) waiting, GPU held by $(gpu_pids)"
    sleep 20
    waited=$((waited + 20))
  done
  return 0
}

run() {  # label policy kv_bytes
  local label=$1 policy=$2 kv=$3
  if [ -d "$R/$label" ]; then echo "SKIP $label"; return 0; fi
  wait_for_gpu "$label" || return 1
  if [ -n "$(gpu_pids)" ]; then echo "RETRY $label"; run "$label" "$policy" "$kv"; return $?; fi
  echo "=== $(date -u +%H:%M:%S) START $label ($policy, kv=$kv) ==="
  timeout 600 "$V" -u run_probe.py --domain long --reservation-policy full \
      --completion-policy "$policy" --cap 32 --gpu-memory-utilization 0.9 \
      --kv-cache-bytes "$kv" --output-dir "$R/$label" > "$R/$label.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) DONE $label exit=$? ==="
  "$V" - "$R/$label/status.json" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print("   ", d.get("status"), "completed", d.get("requests_completed"),
      "preemptions", d.get("actual_preemption_count"), "err", d.get("error"))
PY
}

D0=17181966336   # deficit 0.000 req, usable 8192
D2=16089350144   # deficit 2.035 req, usable 7671  (reproduces the sealed point)
D4=15034482688   # deficit 4.000 req, usable 7168
D6=13960740864   # deficit 6.000 req, usable 6656

# block0: native before rotate at each deficit
for spec in "d0 $D0" "d2 $D2" "d4 $D4" "d6 $D6"; do
  set -- $spec
  run "block0-$1-native" native "$2"
  run "block0-$1-rotate" rotate "$2"
done
# block1: rotate before native, deficits walked in reverse
for spec in "d6 $D6" "d4 $D4" "d2 $D2" "d0 $D0"; do
  set -- $spec
  run "block1-$1-rotate" rotate "$2"
  run "block1-$1-native" native "$2"
done

echo "ALL DONE $(date -u +%H:%M:%S)"
