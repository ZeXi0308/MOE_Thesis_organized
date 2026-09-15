#!/bin/sh
set -eu
'/Users/leandrozhao/Desktop/毕设论文资料/.venv/bin/python' -B docs/ideas/energy_slo/routeslack/experiments/run_routeslack_dry_run.py --output-dir '/Users/leandrozhao/Desktop/毕设论文资料/artifacts/energy_slo_routeslack/20260728_114740' --seed 20260728 --run-tests --include-file /private/tmp/routeslack_tiny_decode_v2.csv=raw/development_tiny_cached_decode_v2.csv --include-file /private/tmp/routeslack_tiny_decode_v2.meta.json=raw/development_tiny_cached_decode_v2.meta.json
