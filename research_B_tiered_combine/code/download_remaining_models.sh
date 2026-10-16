#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while read -r repo name cap; do
    /root/miniconda3/bin/python code/download_model.py --repo "$repo" --name "$name" --max-gib "$cap"
done <<'MODELS'
ibm-granite/granite-3.1-1b-a400m-base granite1b 8
ibm-granite/granite-3.1-3b-a800m-base granite3b 12
allenai/OLMoE-1B-7B-0924 olmoe_base 20
MODELS
