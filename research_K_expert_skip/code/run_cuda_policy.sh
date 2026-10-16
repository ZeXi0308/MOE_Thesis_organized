#!/usr/bin/env bash
set -euo pipefail
EXPERT_SKIP_TEST_CUDA=1 /root/miniconda3/bin/python -m unittest discover -s code/tests -v
/root/miniconda3/bin/python code/microbench_policy.py --output "$1/policy_microbench.json"
