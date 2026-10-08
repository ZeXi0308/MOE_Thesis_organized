#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${D_GPU_UUID:?Set D_GPU_UUID to the selected physical GPU}"
# Same process-local library search path as the already measured parent launcher.
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /root/miniconda3/bin/python -u run_probe.py "$@"
