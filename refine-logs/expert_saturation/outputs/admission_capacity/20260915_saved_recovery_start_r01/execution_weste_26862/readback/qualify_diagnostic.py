"""Actual-action qualification only; no performance threshold or selected repeat."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'pkg'))
from absence_rotation import RotationConfig


def qualify(folder):
    data = {n: json.loads((folder/(n+'.json')).read_text()) for n in
        ('status', 'raw', 'config', 'selective-store', 'offload-events')}
    cfg, selective = data['config'], data['selective-store']
    expected = vars(RotationConfig(min_steps_between_swaps=0))
    errors = []
    if data['status']['status'] != 'COMPLETE' or data['raw']['status'] != 'COMPLETE':
        errors.append('Diagnostic did not complete')
    if cfg['rotation_config'] != expected or selective.get('rotation_config') != expected:
        errors.append('Recorded/applied rotation configuration differs from frozen eager arm')
    if cfg['selective_save'] != 'on' or selective['save'] is not True:
        errors.append('Native saving was not enabled')
    if cfg['fixed_kv_cache_memory_bytes'] != 13960740864 or cfg['offload_gib'] != 16:
        errors.append('Resource budget differs')
    requests = data['raw']['requests']
    if len(requests) != 32 or any(r['status'] != 'completed'
            or len(r['output_token_ids']) != 1024 for r in requests):
        errors.append('Frozen measured request/output budget did not complete')
    events = selective['events']
    ready = {(e['step']-1, e['victim'], e['target']) for e in events
        if e['event'] == 'commit_check' and e['reason'] == 'READY'}
    committed = [e for e in events if e['event'] == 'prepare'
        and (e['step'], e['victim'], e['target']) in ready]
    under_twenty = [dict(previous_prepare_step=a['step'], prepare_step=b['step'],
        interval_steps=b['step']-a['step'], victim=b['victim'], target=b['target'])
        for a, b in zip(committed, committed[1:]) if b['step']-a['step'] < 20]
    if len(committed) != selective['applied_rotations'] or len(committed) < 2:
        errors.append('Committed preparation/action accounting differs')
    accepted = {bool(d['is_store']) for d in data['offload-events']['dispatch'] if d['accepted']}
    completed = {bool(j['is_store']) for e in data['offload-events']['completed_jobs'] for j in e['jobs']}
    if accepted != {False, True} or completed != {False, True}:
        errors.append('Both native save and load were not accepted and completed')
    status = 'INVALID' if errors else ('QUALIFIED' if under_twenty else 'NO_NEW_ACTION')
    return dict(status=status, errors=errors, committed_rotations=len(committed),
        real_actions_forbidden_by_current_cooldown=under_twenty,
        semantics='Action existence on the realized eager state; not an outcome counterfactual. '
                  'Failure/no action retains raw and leaves timing cells UNRUN.')


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    result = qualify(folder)
    with (folder/'action-qualification.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items()
        if k != 'real_actions_forbidden_by_current_cooldown'}))
    raise SystemExit(0 if result['status'] == 'QUALIFIED' else 2)
