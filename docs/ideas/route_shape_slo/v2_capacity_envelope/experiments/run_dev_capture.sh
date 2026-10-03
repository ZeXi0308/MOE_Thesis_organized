#!/usr/bin/env bash
set -euo pipefail

entry_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$entry_dir" rev-parse --show-toplevel)
script_dir="$repo_root/docs/ideas/route_shape_slo/v2_capacity_envelope/experiments"
if [[ -n ${RCE_PYTHON:-} ]]; then
  python_bin=$RCE_PYTHON
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin=python3
fi
run_id=${RCE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
expected_branch=agent/publish-current-moe-code
expected_head=beb08ee4e25dbf93ea3e199db671080c4d3ecea5

if [[ ! "$run_id" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "RCE_RUN_ID must be a UTC timestamp like 20260812T163000Z" >&2
  exit 2
fi

manifest_dir="/tmp/rce-v2-manifests-$run_id"
scratch_dir="/tmp/rce-v2-work-$run_id"
steady_capture="/tmp/bcrd-gate0-smoke-rce-steady-$run_id"
bursty_capture="/tmp/bcrd-gate0-smoke-rce-bursty-$run_id"
artifact_parent="$repo_root/artifacts/route_capacity_envelope/dev"
artifact_dir="$artifact_parent/$run_id"

if [[ -d "$artifact_parent" ]] && find "$artifact_parent" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "a canonical Route Capacity Envelope dev bundle already exists; refusing a second" >&2
  exit 2
fi

for path in \
  "$manifest_dir" \
  "$scratch_dir" \
  "$steady_capture" \
  "$bursty_capture" \
  "$artifact_dir"
do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite existing path: $path" >&2
    exit 2
  fi
done

mkdir -p "$scratch_dir"

run_pipeline() {
  cd "$repo_root"
  current_branch=$(git branch --show-current)
  current_head=$(git rev-parse HEAD)
  [[ "$current_branch" == "$expected_branch" ]] || { echo "expected branch $expected_branch; found $current_branch" >&2; return 2; }
  [[ "$current_head" == "$expected_head" ]] || { echo "expected HEAD $expected_head; found $current_head" >&2; return 2; }
  echo "repo_branch=$current_branch"
  echo "repo_head=$current_head"
  command -v nvidia-smi >/dev/null
  "$python_bin" -c 'import numpy, torch, transformers; assert torch.cuda.is_available(), "CUDA is unavailable"; assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"; assert str(torch.__version__).startswith("2.8.0"), f"expected PyTorch 2.8.0.x, found {torch.__version__}"; assert transformers.__version__ == "4.57.6", f"expected Transformers 4.57.6, found {transformers.__version__}"; name=torch.cuda.get_device_name(0); free,_=torch.cuda.mem_get_info(0); assert "5090" in name, f"expected RTX 5090, found {name}"; assert free >= 24 * 1024**3, f"need 24 GiB free, found {free / 1024**3:.2f} GiB"; print({"gpu": name, "cuda": torch.version.cuda, "free_gib": round(free / 1024**3, 2), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__})'
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
  gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  [[ "$gpu_name" == *"RTX 5090"* ]] || { echo "expected RTX 5090; found $gpu_name" >&2; return 2; }
  (( gpu_free_mib >= 24576 )) || { echo "need 24576 MiB free; found $gpu_free_mib" >&2; return 2; }
  command -v timeout >/dev/null

  set -x
  "$python_bin" "$script_dir/prepare_dev_workloads.py" \
    --spec "$script_dir/olmoe_dev_workload.json" \
    --output-dir "$manifest_dir"

  timeout 10m "$python_bin" "$script_dir/check_telemetry_overhead.py" \
    --workload-manifest "$manifest_dir/olmoe-dev-steady.json" \
    --output "$scratch_dir/telemetry_overhead.json" \
    --requests 8 \
    --decode-steps 8 \
    --warmup-repeats 1 \
    --repeats 3 \
    --max-relative-overhead 0.02 \
    --offline

  timeout 25m "$python_bin" "$script_dir/capture_dev_continuous_decode.py" \
    --workload-manifest "$manifest_dir/olmoe-dev-steady.json" \
    --preregistration docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json \
    --output-dir "$steady_capture" \
    --offline

  timeout 25m "$python_bin" "$script_dir/capture_dev_continuous_decode.py" \
    --workload-manifest "$manifest_dir/olmoe-dev-bursty.json" \
    --preregistration docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json \
    --output-dir "$bursty_capture" \
    --offline

  "$python_bin" "$script_dir/build_capacity_windows.py" \
    --capture-dir "$steady_capture" \
    --capture-dir "$bursty_capture" \
    --num-experts 64 \
    --overhead "$scratch_dir/telemetry_overhead.json" \
    --output "$scratch_dir/windows.csv"

  "$python_bin" "$script_dir/analyze_capacity_signal.py" \
    --windows "$scratch_dir/windows.csv" \
    --config "$script_dir/olmoe_dev_capture.json" \
    --overhead "$scratch_dir/telemetry_overhead.json" \
    --metrics "$scratch_dir/metrics.json" \
    --report "$scratch_dir/report.md"
  set +x

  cp "$script_dir/olmoe_dev_capture.json" "$scratch_dir/config.json"
  cp "$script_dir/run_dev_capture.sh" "$scratch_dir/commands.sh"
  chmod +x "$scratch_dir/commands.sh"
}

run_pipeline 2>&1 | tee "$scratch_dir/run.log"

bundle_stage="$scratch_dir/canonical_bundle"
mkdir "$bundle_stage"
for name in config.json windows.csv metrics.json report.md commands.sh run.log
do
  cp "$scratch_dir/$name" "$bundle_stage/$name"
done

file_count=$(find "$bundle_stage" -maxdepth 1 -type f | wc -l | tr -d ' ')
if [[ "$file_count" != 6 ]]; then
  echo "canonical bundle must contain exactly six files; found $file_count" >&2
  exit 2
fi

mkdir -p "$artifact_parent"
mv "$bundle_stage" "$artifact_dir"
chmod +x "$artifact_dir/commands.sh"

{
  echo "RCE development bundle: $artifact_dir"
  echo "Raw captures: $steady_capture and $bursty_capture"
} | tee -a "$artifact_dir/run.log"
