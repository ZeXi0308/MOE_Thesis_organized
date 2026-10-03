#!/bin/sh
set -eu
/root/miniconda3/bin/python3 -B docs/ideas/energy_slo/routeslack/experiments/run_routeslack_dry_run.py --output-dir /root/autodl-tmp/routeslack_audit_20260728_1443/artifacts/energy_slo_routeslack/20260728_1452_remote_5090 --seed 20260728 --run-tests
