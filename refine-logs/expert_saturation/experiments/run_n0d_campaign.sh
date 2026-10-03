#!/usr/bin/env bash
set -euo pipefail

: "${N0D_SOURCE_ROOT:?set N0D_SOURCE_ROOT to the clean sparse checkout}"
: "${N0D_CAMPAIGN_ROOT:?set N0D_CAMPAIGN_ROOT to a new campaign directory}"
: "${N0D_PYTHON:?set N0D_PYTHON to the CUDA Python executable}"

expected_head=b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a
expected_workload_sha=47babe9d8f875fda3457a68ca83ee7d1274866ebc47013622691d1fc1b556a6d
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd "$N0D_SOURCE_ROOT" && pwd)
campaign_root=$N0D_CAMPAIGN_ROOT
python_bin=$N0D_PYTHON
capture_parent=/root/autodl-tmp/expert-saturation/tmp
capture_dir="$capture_parent/bcrd-gate0-smoke-${campaign_root##*/}"

case "$campaign_root" in
  /root/autodl-tmp/expert-saturation/runs/n0d-*) ;;
  *) echo "N0D_CAMPAIGN_ROOT is outside the authorized N0d run namespace" >&2; exit 2 ;;
esac
if [[ -e "$campaign_root" ]]; then
  echo "refusing to overwrite campaign: $campaign_root" >&2
  exit 2
fi
case "$capture_dir" in
  /root/autodl-tmp/expert-saturation/tmp/bcrd-gate0-smoke-n0d-*) ;;
  *) echo "N0d capture path is outside the authorized temporary namespace" >&2; exit 2 ;;
esac
if [[ -e "$capture_dir" ]]; then
  echo "refusing to overwrite capture: $capture_dir" >&2
  exit 2
fi
if [[ $(/usr/bin/git -C "$source_root" rev-parse HEAD) != "$expected_head" ]]; then
  echo "N0d source checkout HEAD drifted" >&2
  exit 2
fi
if [[ -n $(/usr/bin/git -C "$source_root" status --porcelain --untracked-files=all) ]]; then
  echo "N0d source checkout is not clean" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "N0d Python is not executable" >&2
  exit 2
fi
for required_command in nvidia-smi setsid sha256sum timeout; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required N0d command is missing: $required_command" >&2
    exit 2
  fi
done

current_stage=bootstrap
active_pgid=
campaign_sealed=0
completed_stages=()
last_cleanup_pgid=0
last_cleanup_sent_sigterm=false
last_cleanup_sent_sigkill=false
last_cleanup_group_absent=true

gpu_idle() {
  local rows
  if ! rows=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits); then
    echo "cannot query GPU compute processes" >&2
    return 2
  fi
  if [[ -n ${rows//[[:space:]]/} ]]; then
    echo "GPU compute process overlap detected: $rows" >&2
    return 2
  fi
}

process_group_exists() {
  local pgid=$1
  [[ "$pgid" =~ ^[0-9]+$ ]] && (( pgid > 1 )) || return 1
  kill -0 -- "-$pgid" 2>/dev/null
}

wait_for_process_group_exit() {
  local pgid=$1
  local attempts=$2
  local index
  for ((index = 0; index < attempts; index++)); do
    if ! process_group_exists "$pgid"; then
      return 0
    fi
    sleep 0.1
  done
  ! process_group_exists "$pgid"
}

cleanup_active_group() {
  local pgid=${active_pgid:-}
  last_cleanup_pgid=0
  last_cleanup_sent_sigterm=false
  last_cleanup_sent_sigkill=false
  last_cleanup_group_absent=true
  if [[ -z "$pgid" ]]; then
    return 0
  fi
  if [[ ! "$pgid" =~ ^[0-9]+$ ]] || (( pgid <= 1 )); then
    echo "refusing unsafe process-group cleanup target: $pgid" >&2
    last_cleanup_group_absent=false
    return 2
  fi
  last_cleanup_pgid=$pgid
  if process_group_exists "$pgid"; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    last_cleanup_sent_sigterm=true
  fi
  if ! wait_for_process_group_exit "$pgid" 100; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
    last_cleanup_sent_sigkill=true
  fi
  if ! wait_for_process_group_exit "$pgid" 100; then
    echo "N0d process group survived SIGKILL: $pgid" >&2
    last_cleanup_group_absent=false
    return 2
  fi
  active_pgid=
  return 0
}

record_stage_cleanup() {
  local stage=$1
  local returncode=$2
  local gpu_idle_verified=$3
  local timed_out=false
  local output="$campaign_root/logs/$stage.cleanup.json"
  local temporary="$output.tmp.$$"
  if (( returncode == 124 || returncode == 137 )); then
    timed_out=true
  fi
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite stage cleanup evidence: $output" >&2
    return 2
  fi
  printf '{"gpu_idle_verified":%s,"pgid":%s,"process_group_absent":%s,"returncode":%s,"sent_sigkill":%s,"sent_sigterm":%s,"stage":"%s","timed_out":%s}\n' \
    "$gpu_idle_verified" \
    "$last_cleanup_pgid" \
    "$last_cleanup_group_absent" \
    "$returncode" \
    "$last_cleanup_sent_sigkill" \
    "$last_cleanup_sent_sigterm" \
    "$stage" \
    "$timed_out" > "$temporary"
  mv "$temporary" "$output"
}

run_stage() {
  local stage=$1
  local time_limit=$2
  local log_path=$3
  shift 3
  local returncode
  local cleanup_returncode
  local gpu_idle_returncode
  local gpu_idle_verified=false

  current_stage=$stage
  echo "N0D_STAGE_START=$stage"
  setsid --wait timeout --signal=TERM --kill-after=30s "$time_limit" "$@" \
    > "$log_path" 2>&1 &
  active_pgid=$!
  set +e
  wait "$active_pgid"
  returncode=$?
  cleanup_active_group
  cleanup_returncode=$?
  gpu_idle
  gpu_idle_returncode=$?
  if (( gpu_idle_returncode == 0 )); then
    gpu_idle_verified=true
  fi
  record_stage_cleanup "$stage" "$returncode" "$gpu_idle_verified"
  local evidence_returncode=$?
  set -e

  tail -n 20 "$log_path" || true
  if (( cleanup_returncode != 0 || evidence_returncode != 0 )); then
    echo "N0d stage cleanup evidence failed: $stage" >&2
    return 125
  fi
  if (( gpu_idle_returncode != 0 )); then
    echo "N0d stage did not release the experiment GPU: $stage" >&2
    return 126
  fi
  if (( returncode != 0 )); then
    echo "N0d stage failed without retry: $stage exit=$returncode" >&2
    return "$returncode"
  fi
  completed_stages+=("$stage")
  echo "N0D_STAGE_COMPLETE=$stage"
}

write_campaign_aborted() {
  local exit_code=$1
  local cleanup_returncode=$2
  local gpu_idle_verified=$3
  local completed_csv=
  if (( ${#completed_stages[@]} > 0 )); then
    completed_csv=$(IFS=,; printf '%s' "${completed_stages[*]}")
  fi
  N0D_ABORT_PATH="$campaign_root/CAMPAIGN_ABORTED.json" \
  N0D_COMPLETE_PATH="$campaign_root/CAMPAIGN_COMPLETE.json" \
  N0D_ABORT_STAGE="$current_stage" \
  N0D_ABORT_EXIT_CODE="$exit_code" \
  N0D_ABORT_COMPLETED="$completed_csv" \
  N0D_ABORT_CLEANUP_RC="$cleanup_returncode" \
  N0D_ABORT_CLEANUP_PGID="$last_cleanup_pgid" \
  N0D_ABORT_GROUP_ABSENT="$last_cleanup_group_absent" \
  N0D_ABORT_GPU_IDLE="$gpu_idle_verified" \
  "$python_bin" -c '
import json, os
from pathlib import Path

output = Path(os.environ["N0D_ABORT_PATH"])
if Path(os.environ["N0D_COMPLETE_PATH"]).exists() or output.exists():
    raise SystemExit(0)
payload = {
    "schema": "n0d-campaign-aborted-v1",
    "status": "CAMPAIGN_ABORTED",
    "failed_stage": os.environ["N0D_ABORT_STAGE"],
    "exit_code": int(os.environ["N0D_ABORT_EXIT_CODE"]),
    "completed_stages": [
        value for value in os.environ["N0D_ABORT_COMPLETED"].split(",") if value
    ],
    "retry_performed": False,
    "process_cleanup": {
        "cleanup_returncode": int(os.environ["N0D_ABORT_CLEANUP_RC"]),
        "pgid": int(os.environ["N0D_ABORT_CLEANUP_PGID"]),
        "process_group_absent": os.environ["N0D_ABORT_GROUP_ABSENT"] == "true",
        "gpu_idle_verified": os.environ["N0D_ABORT_GPU_IDLE"] == "true",
    },
}
temporary = output.with_name(".{0}.{1}.tmp".format(output.name, os.getpid()))
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.link(str(temporary), str(output))
temporary.unlink()
'
}

campaign_exit() {
  local original_exit=$1
  local final_exit=$original_exit
  local cleanup_returncode
  local gpu_idle_returncode
  local gpu_idle_verified=false
  trap - EXIT INT TERM
  if (( campaign_sealed == 1 )); then
    exit "$original_exit"
  fi
  if (( final_exit == 0 )); then
    final_exit=2
  fi
  set +e
  cleanup_active_group
  cleanup_returncode=$?
  gpu_idle
  gpu_idle_returncode=$?
  if (( gpu_idle_returncode == 0 )); then
    gpu_idle_verified=true
  fi
  write_campaign_aborted "$final_exit" "$cleanup_returncode" "$gpu_idle_verified"
  set -e
  exit "$final_exit"
}

campaign_signal() {
  local exit_code=$1
  exit "$exit_code"
}

mkdir -p "$campaign_root/frozen" "$campaign_root/bundles" "$campaign_root/logs"
install -d -m 0755 "$capture_parent"
trap 'campaign_exit $?' EXIT
trap 'campaign_signal 130' INT
trap 'campaign_signal 143' TERM

current_stage=freeze-controls
cp "$script_dir/run_n0d_campaign.sh" "$campaign_root/frozen/"
cp "$script_dir/run_n0d_matched_router_gate.py" "$campaign_root/frozen/"
cp "$script_dir/evaluate_n0d_matched_router_gate.py" "$campaign_root/frozen/"
cp "$script_dir/n0d_capture_contract.py" "$campaign_root/frozen/"
cp "$script_dir/N0D_MATCHED_ROUTER_GATE.md" "$campaign_root/frozen/"
(
  cd "$campaign_root/frozen"
  sha256sum \
    run_n0d_campaign.sh \
    run_n0d_matched_router_gate.py \
    evaluate_n0d_matched_router_gate.py \
    n0d_capture_contract.py \
    N0D_MATCHED_ROUTER_GATE.md \
    > control-files.sha256
)

manifest_dir="$campaign_root/frozen/manifests"
workload="$manifest_dir/olmoe-dev-steady.json"
prepare="$source_root/docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/prepare_dev_workloads.py"
spec="$source_root/docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/olmoe_dev_workload.json"
capture="$source_root/docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/capture_dev_continuous_decode.py"
prereg="$source_root/docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json"
runner="$campaign_root/frozen/run_n0d_matched_router_gate.py"
evaluator="$campaign_root/frozen/evaluate_n0d_matched_router_gate.py"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$capture_parent"
unset PYTHONPYCACHEPREFIX

gpu_idle
run_stage prepare 5m "$campaign_root/logs/prepare.log" \
  "$python_bin" "$prepare" --spec "$spec" --output-dir "$manifest_dir"
current_stage=verify-workload
echo "$expected_workload_sha  $workload" | sha256sum -c -

run_stage capture 30m "$campaign_root/logs/capture.log" \
  "$python_bin" "$capture" \
  --workload-manifest "$workload" \
  --preregistration "$prereg" \
  --output-dir "$capture_dir" \
  --offline

# Do not spend the three measured model processes when the fresh, sealed
# source capture no longer contains the phenomenon this localization Gate
# conditions on.
run_stage verify-capture 1m "$campaign_root/logs/verify-capture.log" \
  "$python_bin" "$campaign_root/frozen/n0d_capture_contract.py" \
  --capture-dir "$capture_dir"

for process_repeat in 0 1 2; do
  output="$campaign_root/bundles/process-$process_repeat.json"
  run_stage "process-$process_repeat" 30m \
    "$campaign_root/logs/process-$process_repeat.log" \
    "$python_bin" "$runner" \
    --source-root "$source_root" \
    --capture-dir "$capture_dir" \
    --workload-manifest "$workload" \
    --process-repeat "$process_repeat" \
    --output "$output"
done

run_stage evaluate 5m "$campaign_root/logs/evaluate.log" \
  "$python_bin" "$evaluator" \
  --input "$campaign_root/bundles/process-0.json" \
  --input "$campaign_root/bundles/process-1.json" \
  --input "$campaign_root/bundles/process-2.json" \
  --capture-dir "$capture_dir" \
  --output "$campaign_root/n0d-verdict.json"

current_stage=seal-campaign
expected_completed=prepare,capture,verify-capture,process-0,process-1,process-2,evaluate
observed_completed=$(IFS=,; printf '%s' "${completed_stages[*]}")
if [[ "$observed_completed" != "$expected_completed" ]]; then
  echo "N0d completed-stage set did not close: $observed_completed" >&2
  exit 2
fi
(
  cd "$campaign_root"
  find . -type f \
    ! -name CAMPAIGN_FILES.sha256 \
    ! -name CAMPAIGN_COMPLETE.json \
    ! -name CAMPAIGN_ABORTED.json \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum > CAMPAIGN_FILES.sha256
)
(
  cd "$campaign_root/frozen"
  sha256sum -c control-files.sha256
)
(
  cd "$campaign_root"
  sha256sum -c CAMPAIGN_FILES.sha256
)
gpu_idle

N0D_CAMPAIGN_ROOT_FOR_SEAL="$campaign_root" \
N0D_CAPTURE_DIR_FOR_SEAL="$capture_dir" \
"$python_bin" -c '
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["N0D_CAMPAIGN_ROOT_FOR_SEAL"])
capture = Path(os.environ["N0D_CAPTURE_DIR_FOR_SEAL"])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

stages = [
    "prepare",
    "capture",
    "verify-capture",
    "process-0",
    "process-1",
    "process-2",
    "evaluate",
]
cleanup = {}
for stage in stages:
    path = root / "logs" / (stage + ".cleanup.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("stage") != stage
        or value.get("returncode") != 0
        or value.get("process_group_absent") is not True
        or value.get("gpu_idle_verified") is not True
    ):
        raise SystemExit("invalid cleanup evidence for " + stage)
    cleanup[stage] = sha(path)

required = [
    capture / "CAPTURE_COMPLETE.json",
    root / "bundles" / "process-0.json",
    root / "bundles" / "process-1.json",
    root / "bundles" / "process-2.json",
    root / "n0d-verdict.json",
    root / "CAMPAIGN_FILES.sha256",
    root / "frozen" / "control-files.sha256",
]
if any(not path.is_file() for path in required):
    raise SystemExit("campaign sealing input is missing")
verdict = json.loads((root / "n0d-verdict.json").read_text(encoding="utf-8"))
if (
    verdict.get("schema") != "n0d-matched-router-evaluation-v1"
    or not isinstance(verdict.get("status"), str)
    or verdict.get("capacity_claim_authorized") is not False
    or verdict.get("action_oracle_authorized") is not False
    or verdict.get("controller_authorized") is not False
    or verdict.get("method_go_authorized") is not False
):
    raise SystemExit("campaign verdict contract is invalid")

payload = {
    "schema": "n0d-campaign-complete-v1",
    "status": "CAMPAIGN_COMPLETE",
    "claim_ceiling": "CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY",
    "completed_stages": stages,
    "retry_performed": False,
    "gpu_idle_after_campaign": True,
    "campaign_files_manifest_sha256": sha(root / "CAMPAIGN_FILES.sha256"),
    "control_files_manifest_sha256": sha(root / "frozen" / "control-files.sha256"),
    "capture_dir": str(capture),
    "capture_complete_sha256": sha(capture / "CAPTURE_COMPLETE.json"),
    "process_outputs_sha256": {
        "process-0": sha(root / "bundles" / "process-0.json"),
        "process-1": sha(root / "bundles" / "process-1.json"),
        "process-2": sha(root / "bundles" / "process-2.json"),
    },
    "verdict_sha256": sha(root / "n0d-verdict.json"),
    "verdict": {
        "status": verdict["status"],
        "structurally_valid": verdict.get("structurally_valid"),
        "failure_category": verdict.get("failure_category"),
    },
    "cleanup_evidence_sha256": cleanup,
}
output = root / "CAMPAIGN_COMPLETE.json"
if output.exists():
    raise SystemExit("refusing to overwrite campaign completion sentinel")
temporary = output.with_name(".{0}.{1}.tmp".format(output.name, os.getpid()))
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(str(temporary), str(output))
'

# The completion sentinel is the final campaign filesystem mutation.
campaign_sealed=1
echo "N0D_CAMPAIGN_COMPLETE=$campaign_root"
