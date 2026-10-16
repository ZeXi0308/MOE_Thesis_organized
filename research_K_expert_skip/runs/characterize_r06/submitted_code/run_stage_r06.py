"""Continue independent characterization stages under one outer GPU lock."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
out = Path(sys.argv[1])
py = sys.executable
model = '/root/autodl-tmp/moe-research-20261002/model'
data = root / 'datasets/wikitext2_serving_r01'
state = {'scope': 'exploratory characterization, not final confirmation', 'stages': []}


def save():
    p = out / 'stages.json'
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(state, indent=2))
    q.replace(p)


def run(name, command, env=None):
    row = {'name': name, 'command': list(map(str, command)), 'started_unix': time.time(),
           'tuned_config_folder': (env or os.environ).get('VLLM_TUNED_CONFIG_FOLDER')}
    state['stages'].append(row)
    save()
    with (out / (name + '.log')).open('w') as log:
        p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    row.update(exit_code=p.returncode, ended_unix=time.time())
    save()
    print(json.dumps(row), flush=True)
    return p.returncode == 0


traces = sorted((root / 'runs/graph_resume_r05/bench_smoke').glob('trace_native_b*/snapshot.json'))
assert len(traces) == 7
tuned = run('native_tuning', [py, 'code/tune_moe.py', '--model', model,
    '--traces', *map(str, traces), '--output-dir', str(out / 'native_tuning'),
    '--batches', '1', '2', '4', '8', '16', '32', '64'])
env = os.environ.copy()
if tuned:
    env['VLLM_TUNED_CONFIG_FOLDER'] = str((out / 'native_tuning/tuned_configs').resolve())

dense_ok = run('graph_dense', [py, 'code/smoke_graph.py', '--model', model,
    '--output', str(out / 'graph_dense.json'), '--spec-json', '{"mode":"dense","backend":"triton"}'], env)
marginal_ok = False
if dense_ok:
    marginal_ok = run('graph_marginal', [py, 'code/smoke_graph.py', '--model', model,
        '--output', str(out / 'graph_marginal.json')], env)

pareto = json.loads(Path('configs/quality_pareto.json').read_text())
if isinstance(pareto, dict):
    pareto = pareto['arms']
t1 = json.loads(Path('configs/drop_rank_gate.json').read_text())
combined = pareto + [a for a in t1['arms'] if a['spec'] and a['spec']['mode'] == 'rank_gate']
assert len({a['name'] for a in combined}) == len(combined)
combined_path = out / 'quality_combined_b8_arms.json'
combined_path.write_text(json.dumps({'arms': combined, 't1_semantics': t1['semantics']}, indent=2))
for batch in (8, 1, 32):
    run(f'quality_b{batch}', [py, 'code/quality_decode.py', '--model', model,
        '--articles', str(data / 'validation_quality32.jsonl'),
        '--output', str(out / f'quality_b{batch}.json'), '--batch', str(batch),
        '--tokens', '32', '--prefix-tokens', '128', '--limit-docs', '32',
        '--arms-file', str(combined_path) if batch == 8 else 'configs/quality_pareto.json',
        '--gpu-memory-utilization', '.35'], env)

# FP8 has its own dtype-specific kernel configurations, not this BF16 tuning.
run('fp8_smoke', ['bash', 'code/run_fp8_smoke.sh', str(out / 'fp8_smoke')])
if tuned and dense_ok and marginal_ok:
    run('graph_pilot', [py, 'code/bench_graph.py', '--model', model,
        '--prompts', str(data / 'validation_bench64.jsonl'), '--output-dir', str(out / 'graph_pilot'),
        '--arms-file', 'configs/default_arms.json', '--arms', 'native', 'global_j4', 'marginal_high',
        '--batches', '1', '8', '32', '--repeats', '1', '--newtokens', '128',
        '--prompt-tokens', '128', '--warmup-seconds', '1', '--no-energy'], env)
state['completed_unix'] = time.time()
state['status'] = 'COMPLETE' if all(s['exit_code'] == 0 for s in state['stages']) else 'PARTIAL_FAILURE'
save()
sys.exit(0 if state['status'] == 'COMPLETE' else 1)
