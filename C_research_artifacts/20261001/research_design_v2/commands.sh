#!/bin/sh
# CPU-only entry points. No SSH, GPU initialization, or experiment launch.
set -eu
report_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "${1:-summary}" in
  summary)
    python3 - "$report_dir" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
r=json.loads((p/'mechanism_probe/result.json').read_text())
print('CPU verdict:', r['verdict'])
print('Generated states:', r['generated']['exhaustive_states']+r['generated']['random_states'])
print('Observed pure-decode states:', sum(x['all_decode_calls'] for x in r['observed'].values()))
f=json.loads((p/'fresh_analysis_recovered.json').read_text())
for c in f['cells']:
 m=c['metrics']
 print(c['label'], 'completed=',m['completed_requests'], 'goodput20/4=',round(m['goodput_20_4_requests_s'],6))
print('New GPU campaign: UNRUN in this session')
PY
    ;;
  reproduce-cpu)
    result_dir=$(mktemp -d /private/tmp/c-retirement-packing-repro.XXXXXX)
    python3 -B "$report_dir/mechanism_probe/probe.py" --output "$result_dir/result.json"
    python3 -B "$report_dir/prior_art_components.py" --out "$result_dir/prior_art_component_result.json"
    printf 'New CPU outputs: %s\n' "$result_dir"
    ;;
  prior-component)
    python3 -B "$report_dir/prior_art_components.py"
    ;;
  reanalyze-e0)
    # E0 is already complete. This reads raw files and writes a NEW result only.
    source_dir=/private/tmp/moe-research-c-20260929-v2/refine-logs/expert_saturation/experiments/admission_capacity/20260929_c_baseline_delivery
    result_dir=$(mktemp -d /private/tmp/c-fresh-analysis-repro.XXXXXX)
    python3 -B "$source_dir/C_NATIVE_RETIREMENT_FRESH_ANALYZE_V1.py" \
      --root "$report_dir/../c-native-retirement-fresh-v1" \
      --output "$result_dir/analysis.json"
    printf 'New E0 derived output: %s\n' "$result_dir/analysis.json"
    ;;
  *)
    printf 'Usage: sh commands.sh [summary|reproduce-cpu|prior-component|reanalyze-e0]\n' >&2
    exit 2
    ;;
esac
