#!/usr/bin/env bash
# One engine start, two measurements.
#
# The expert-union route collection and the envelope model's P1-P3 static
# sweep both need the same pinned engine, the same frozen inputs and the same
# warmup discipline. Loading the model twice would double GPU time and, worse,
# would compare two measurements taken under different cache and clock state.
#
# Order matters and is fixed here:
#   1. route collection first, because it is the prerequisite the repository
#      named and the only measurement whose verdict is already frozen;
#   2. the static cap sweep second, so a failure in the (newer, less certain)
#      sweep cannot cost us the route data.
#
# Every episode is retained regardless of outcome. Nothing is overwritten.
set -u

V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin
WORK=/root/autodl-tmp/expert-union
OUT=${1:?usage: run_session.sh <output-root>}
DOMAIN=${2:-short}
CAP=${3:-32}

export HF_HOME=/root/autodl-tmp/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
cd "$WORK" || exit 90

{
  echo "host=$(hostname) date=$(date -Is)"
  echo "domain=$DOMAIN cap=$CAP"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  $V/python -c "import vllm,torch,transformers;print('stack',vllm.__version__,torch.__version__,transformers.__version__)"
  sha256sum ./*.py
} > "$OUT/session-env.txt" 2>&1

# Refuse to start if the GPU is not ours: a co-tenant process would silently
# change every timing number and invalidate the staircase comparison.
GPU_STATE=$(nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv,noheader 2>&1)
GPU_RC=$?
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>&1)
PROCESS_RC=$?
{
  printf 'UTC=%s caller_pid=%s stage=before_model_load\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$"
  printf 'gpu_query_rc=%s\n%s\nprocess_query_rc=%s\n%s\n' "$GPU_RC" "$GPU_STATE" "$PROCESS_RC" "$GPU_PROCESSES"
} >> "$OUT/gpu-checks.log"
if [ "$GPU_RC" -ne 0 ] || [ "$PROCESS_RC" -ne 0 ] || [ -z "${GPU_STATE//[[:space:]]/}" ]; then
  echo "ABORT: GPU state query failed; retained in gpu-checks.log" | tee -a "$OUT/session-env.txt"
  exit 91
fi
if [ -n "${GPU_PROCESSES//[[:space:]]/}" ]; then
  echo "ABORT: foreign GPU compute process(es); retained in gpu-checks.log" | tee -a "$OUT/session-env.txt"
  exit 91
fi

echo "=== [1/2] expert-union route collection ($(date +%T)) ===" | tee -a "$OUT/session.log"
$V/python run_expert_union.py \
  --domain "$DOMAIN" --cap "$CAP" \
  --prepared-dir "$WORK/inputs_preparation/prepared/$DOMAIN" \
  --output-dir "$OUT/union" \
  --max-steps 400 --output-tokens 256 \
  >> "$OUT/union.stdout.log" 2>> "$OUT/union.stderr.log"
UNION_RC=$?
echo "union exit=$UNION_RC" | tee -a "$OUT/session.log"

# The route measurement is the frozen prerequisite; report it immediately so a
# later failure cannot bury it.
if [ -f "$OUT/union/model_shape.json" ]; then
  echo "--- model_shape.json ---" | tee -a "$OUT/session.log"
  cat "$OUT/union/model_shape.json" | tee -a "$OUT/session.log"
fi
if [ -f "$OUT/union/union_summary.json" ]; then
  $V/python -c "
import json,sys
d=json.load(open('$OUT/union/union_summary.json'))
v=d['verdict']
print('VERDICT', v['verdict'])
print('max layer median idle', round(v['max_layer_median_idle_fraction'],4))
print('layers never saturating', v.get('n_layers_never_saturating'))
print('recorded steps', d['n_recorded_steps'])
print('discipline', json.dumps(d.get('collection_discipline',{})))
" 2>&1 | tee -a "$OUT/session.log"
fi

echo "=== [2/2] static cap sweep for P1-P3 ($(date +%T)) ===" | tee -a "$OUT/session.log"
echo "SKIPPED: driver not yet deployed; route data is the frozen prerequisite" \
  | tee -a "$OUT/session.log"

echo "done rc=$UNION_RC $(date -Is)" | tee -a "$OUT/session.log"
exit "$UNION_RC"
