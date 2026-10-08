#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "$0")" && pwd)
test "$#" -ge 1 || { echo 'Usage: bash launch.sh NEW_OUTPUT_DIRECTORY [--profile ...] [--selection ...]' >&2; exit 2; }
test ! -e "$1" || { echo 'Refusing to reuse output directory' >&2; exit 2; }
task_output=$1
shift
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib
export XDG_CACHE_HOME="$task_root/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$task_root/cache/torchinductor"
export TRITON_CACHE_DIR="$task_root/cache/triton"
export VLLM_CACHE_ROOT="$task_root/cache/vllm"
export PYTHONDONTWRITEBYTECODE=1
# Also repair fresh Python interpreters started by vLLM's model registry/workers.
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
exec /root/miniconda3/bin/python -u "$task_root/run.py" --output "$task_output" "$@"
