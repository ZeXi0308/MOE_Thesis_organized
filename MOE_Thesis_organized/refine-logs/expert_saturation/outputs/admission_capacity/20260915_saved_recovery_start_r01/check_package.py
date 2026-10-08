"""CPU-only checks for the changed gate, action qualification and comparison contract."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'pkg'))
from absence_rotation import AbsenceRotation, RequestView, RotationConfig
import analyze_saved_recovery_start as analysis
from qualify_diagnostic import qualify

for p in root.rglob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
subprocess.run(['bash', '-n', str(root/'pkg/run.sh')], check=True)
subprocess.run([sys.executable, str(root/'pkg/run_probe.py'), '--help'],
    check=True, stdout=subprocess.DEVNULL)
assert RotationConfig().min_steps_between_swaps == 20


def decide(cooldown=20, *, step=100, absent=60, resident=60, free=0):
    tracker = AbsenceRotation(config=RotationConfig(min_steps_between_swaps=cooldown),
        absent_since={'waiting': absent}, resident_since={'running': resident},
        last_swap_step=95, victim_order='most_output')
    return tracker.decide(step, [RequestView('running', 3091, 3072, 4096, 20)],
        ['waiting'], free, {'waiting': 194})


assert decide().reason == 'swap cooldown'
assert decide(0).action == 'rotate'
for kwargs in (dict(absent=90), dict(resident=90), dict(free=194)):
    assert decide(0, **kwargs).action == 'noop'
assert vars(decide(20, step=115)) == vars(decide(0, step=115))

# Only successful committed preparations qualify a realized sub-20-step action.
cfg = dict(rotation_config=vars(RotationConfig(min_steps_between_swaps=0)),
    selective_save='on', fixed_kv_cache_memory_bytes=13960740864, offload_gib=16)
fixture = {'status': {'status': 'COMPLETE'}, 'config': cfg,
    'raw': {'status': 'COMPLETE', 'requests': [dict(status='completed',
        output_token_ids=[0]*1024) for _ in range(32)]},
    'selective-store': dict(save=True, rotation_config=cfg['rotation_config'], applied_rotations=2,
        events=[dict(event='prepare', step=s, victim='v', target='t') for s in (100, 105)]
        +[dict(event='commit_check', step=s, victim='v', target='t', reason='READY') for s in (101, 106)]),
    'offload-events': dict(dispatch=[dict(is_store=v, accepted=True) for v in (False, True)],
        completed_jobs=[dict(jobs=[dict(is_store=v) for v in (False, True)])])}
with tempfile.TemporaryDirectory(prefix='saved-start-check-') as tmp:
    folder = Path(tmp)
    def run_fixture(value):
        for name, data in value.items():
            (folder/(name+'.json')).write_text(json.dumps(data))
        return qualify(folder)
    assert run_fixture(fixture)['status'] == 'QUALIFIED'
    altered = copy.deepcopy(fixture)
    altered['selective-store']['events'][1]['step'] = 125
    altered['selective-store']['events'][3]['step'] = 126
    assert run_fixture(altered)['status'] == 'NO_NEW_ACTION'
    altered['selective-store']['rotation_config']['min_absence_steps'] = 0
    assert run_fixture(altered)['status'] == 'INVALID'

# Mixed save arms and any other budget/config change are rejected before metrics.
simple_request = dict(request_id='r', document_id='d', prompt_sha256='p', arrival_s=0,
    max_output=1024, output_sha256='o', completion_latency_s=2,
    max_engine_return_gap_s=0.1, ttft_s=0.1)
requests = dict(requests=[simple_request], **{k: 1 for k in analysis.METRICS})
cells = {}
for name in analysis.NAMES:
    config = copy.deepcopy(cfg)
    config['rotation_config']['min_steps_between_swaps'] = 20 if name.endswith('current') else 0
    cells[name] = dict(comparable=not name.startswith('diagnostic'), config=config, requests=requests)
original = analysis.cell
analysis.cell = lambda p, _: cells[p.name]
try:
    assert all(p['status'] == 'COMPLETE' for p in analysis.analyze(Path('/unrun'))['performance_comparisons'])
    cells['block0-eager']['config']['offload_gib'] = 8
    assert analysis.analyze(Path('/unrun'))['performance_comparisons'][0]['status'] == 'NOT_COMPARABLE'
    cells['block0-eager']['config']['offload_gib'] = 16
    cells['block0-eager']['config']['selective_save'] = 'off'
    assert analysis.analyze(Path('/unrun'))['performance_comparisons'][0]['status'] == 'NOT_COMPARABLE'
finally:
    analysis.cell = original
print('PASS: AST/shell/CLI; changed cooldown; unchanged guards; realized-action gate; comparison config isolation')
