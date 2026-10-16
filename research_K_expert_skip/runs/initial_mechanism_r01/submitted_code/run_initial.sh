#!/usr/bin/env bash
set -euo pipefail
python=/root/miniconda3/bin/python
"$python" -m unittest discover -s code/tests -v
"$python" code/verify_mechanism.py --output "$1/mechanism.json"
"$python" code/smoke_model.py --model /root/autodl-tmp/moe-research-20261002/model --output "$1/model_smoke.json"
