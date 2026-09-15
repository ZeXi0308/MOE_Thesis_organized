#!/usr/bin/env bash
set -euo pipefail

# Exact GPU commands are retained verbatim in the canonical source bundles:
#   artifacts/longrun_A_execution_conformance/20260812T204037Z/commands.sh
#   artifacts/longrun_A_execution_conformance/prevalence/20260812T205319Z/commands.sh

# Read-only manifest verification used for this summary.
cd artifacts/longrun_A_execution_conformance/20260812T204037Z
jq -r '.files | to_entries[] | "\(.value.sha256)  \(.key)"' RUN_COMPLETE.json | shasum -a 256 -c -

cd ../prevalence/20260812T205319Z
jq -r '.files | to_entries[] | "\(.value.sha256)  \(.key)"' RUN_COMPLETE.json | shasum -a 256 -c -

cd ../../../..
find artifacts/longrun_A_execution_conformance/20260812T204037Z \
  artifacts/longrun_A_execution_conformance/prevalence/20260812T205319Z \
  idea-stage/resurrection -type f -name '*.json' -print0 | xargs -0 -n1 jq empty

PYTHONPYCACHEPREFIX=/tmp/resurrection-pycache ./.venv/bin/python -m py_compile \
  idea-stage/resurrection/experiments/run_source_localization.py \
  idea-stage/resurrection/experiments/test_source_localization.py

PYTHONPYCACHEPREFIX=/tmp/resurrection-pycache ./.venv/bin/python -m unittest -v \
  idea-stage/resurrection/experiments/test_source_localization.py
