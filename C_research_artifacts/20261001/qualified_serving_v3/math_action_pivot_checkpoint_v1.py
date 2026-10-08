#!/usr/bin/env python3
"""Compact, evidence-linked development summary; never includes full token traces."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ARMS = {
    'native512_B2048': 'math_scale512_analysis_v1.json',
    'native1024_B2048': 'math_scale1024_analysis_v1.json',
    'native768_B2048': 'math_recovery_native768_analysis_v1.json',
    'restore_fifo1024_B2048': 'math_recovery_fifo_analysis_v1.json',
    'native1024_B2048_repeat': 'math_recovery_native1024repeat_analysis_v1.json',
    'native768_B4096': 'math_batch4096_native768_analysis_v1.json',
    'restore_runway1024_B2048': 'math_runway_analysis_v1.json',
    'native896_B2048': 'math_controls_native896_analysis_v1.json',
    'native1024_B4096': 'math_controls_native1024b4096_analysis_v1.json',
    'restore_runway1024_B4096': 'math_runway4096_analysis_v1.json',
    **{'fixed_margin%d_B4096' % m: 'math_fixedmargin%d_analysis_v1.json' % m for m in (32, 48, 64)},
}

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    result = dict(schema='math-action-pivot-checkpoint-v1', created_unix_s=time.time(),
        goal_status='ACTIVE_NOT_ACHIEVED', paper_status='NO_PAPER_GO',
        scope='Exploration on the first1024 GSM8K test requests, Qwen2.5-7B-Instruct, full original arrivals. No held-out result.',
        fixed=dict(kv_bytes=8589934592, APC=True, max_output_tokens=1024,
                   temperature=0, seed=20260905, batch_invariance=False,
                   input_sha256='513ae9ac6dff4e36693c07c1c37cd9d6d6101612beda5aad59bb33994104e13a'),
        metrics_scope='Instrumented host timings; diagnostic TTFT20s / maximum successive emitted-token host gap4s. Not device ITL, business SLO, or steady-state capacity.',
        arms={}, unavailable_analyses=[],
        evidence_references=['math_runway_common_prefix_v1.json', 'math_fifo_common_prefix_v1.json',
            'math_runway_mechanism_comparison_v1.json', 'math_runway_margin_action_check_v1.json',
            'math_runway_nearest_methods_v1.json', 'math_native4096_vsrunway4096_pair_v1.json',
            'restore_runway_policy_v2.py', 'restore_fixed_margin_policy_v1.py'],
        interpretation_limits=[
            'Every request remains in performance and quality denominators, including errors and capped generations.',
            'Same seed does not imply identical natural outputs across scheduling policies; report output length and paired identical-output sensitivity.',
            'The B2048 FIFO makespan improvement is confounded by a nonpreempted capped tail request shortening; most requests worsened.',
            'Tuned native B4096 already beats the dynamic B2048 policy; cross-configuration improvement is not a policy benefit.',
            'The matched observable prestate supports a first-action witness, not identical GPU state or all later actions.',
            'Reduced preemptions or recomputed token positions are not sufficient evidence of terminal performance benefit.',
            'The one-step gate has no persistent reservation and does not guarantee execution progress.',
            'Closest methods already cover reservation, recovery costs and protected progress; novelty is unresolved.',
            'Single development runs do not establish repeatability, uncertainty, held-out generalization or CCF-C publication readiness.'])
    for label, filename in ARMS.items():
        p = base / filename
        if not p.exists():
            result['unavailable_analyses'].append(filename)
            continue
        with p.open() as f:
            data = json.load(f)
        result['arms'][label] = dict(analysis=filename, sha256=digest(p),
            **{k: data.get(k) for k in ('integrity', 'issues', 'counts', 'episode', 'timing', 'pressure_counts')},
            scheduled_work_totals=data.get('attribution', {}).get('totals'))
        del data
    with Path(args.output).open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=args.output, arms=len(result['arms']),
                         unavailable=result['unavailable_analyses'])))

if __name__ == '__main__':
    main()
