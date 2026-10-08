"""Check all reservation receipts and actual preemption windows; no ready-time inference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from natural_pause import diagnose

ROOT = Path(__file__).parent
NAMES = ('00_swap16', '01_headroom', '02_reserve16', '03_reserve16', '04_headroom', '05_swap16')
load = lambda p: json.loads(p.read_text())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results_kv_reserve_r01')
    p.add_argument('--source', type=Path, default=ROOT / 'kv_reserve_src/restore_reserve.py')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    source = args.source.read_bytes(); source_sha = hashlib.sha256(source).hexdigest()
    fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == 'allocate')
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'original')
    forwarded = {k.arg: ast.unparse(k.value) for k in call.keywords}
    unchanged = (ast.unparse(call.args[1]) == 'num_new_tokens' and forwarded['num_external_computed_tokens'] == 'num_external_computed_tokens'
        and not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in ('num_new_tokens', 'num_external_computed_tokens') for n in ast.walk(fn)))
    cells = {}
    for name in NAMES:
        directory = args.results / name
        raw, receipt, config = [load(directory / f) for f in ('raw.json', 'restore_reservation.json', 'config.json')]
        if raw['status'] != 'COMPLETE' or load(directory / 'status.json')['status'] != 'COMPLETE':
            p.error(f'No report written: {name} is not COMPLETE')
        resources = load(directory / 'resources.json'); block = resources['block_size']
        pools = [resources['num_gpu_blocks'], resources['pool_num_gpu_blocks'], load(directory / 'measurement_resources.json')['kv_offload']['gpu_blocks'], load(directory / 'kv_resources_after_measurement.json')['gpu_blocks']]
        enabled = 'reserve16' in name
        pauses = diagnose(raw)['preemptions']; actions = []
        for a in receipt['actions']:
            sid = raw['internal_to_source'].get(a['request_id'])
            matches = [i for i, q in enumerate(pauses) if q['request_id'] == sid]
            ordinal = a['num_preemptions']; index = matches[ordinal - 1] if 0 < ordinal <= len(matches) else None
            prefix, tail = a['cached_prefix_tokens'], a['known_tail_tokens']
            checks = dict(source_and_preemption_found=index is not None, positive_preemption_count=ordinal > 0,
                aligned_positive_prefix=prefix > 0 and prefix % block == 0, one_block_tail=0 < tail <= block,
                same_block_size=a['block_size'] == block, zero_new_tokens=a['num_new_tokens'] == 0,
                external_equals_prefix=a['external_tokens'] == prefix, lookahead_zero_to_one=a['lookahead_before'] == 0 and a['lookahead_after'] == 1,
                allocated_prefix_plus_one=a['total_request_blocks_after'] == [prefix // block + 1] if a['outcome'] == 'allocated' else a['total_request_blocks_after'] is None,
                valid_outcome=a['outcome'] in ('allocated', 'denied'))
            actions.append(dict(receipt=a, source_request_id=sid, actual_preemption_window_index=index, receipt_condition_checks=checks))
        checks = dict(enabled_flag=config['restore_reserve_block'] == receipt['enabled'] == enabled,
            installed_flag=receipt['installed'] == enabled, fixed_512_pool=all(x == 512 for x in pools),
            helper_source_matches=load(directory / 'environment.json')['sources']['restore_reserve.py'] == source_sha,
            source_forwards_original_external_and_new_tokens=unchanged,
            disabled_no_actions_or_calls=enabled or (not receipt['actions'] and all(receipt[k] == 0 for k in ('calls', 'eligible', 'allocated', 'denied'))),
            receipt_counts=receipt['eligible'] == len(actions) == receipt['allocated'] + receipt['denied'] and all(receipt[k] == sum(a['receipt']['outcome'] == k for a in actions) for k in ('allocated', 'denied')),
            all_recorded_action_conditions=all(all(a['receipt_condition_checks'].values()) for a in actions))
        cells[name] = dict(checks=checks, pool_sizes=pools, counts={k: receipt[k] for k in ('calls', 'eligible', 'allocated', 'denied')},
            unique_action_requests=sorted({a['source_request_id'] for a in actions if a['source_request_id']}), actions=actions,
            actual_preemption_windows=pauses, receipt_step_to_capture_alignment='UNVERIFIED: no shared raw sched_step_seq field',
            ready_to_schedule_wait_s=None)
    report = dict(scope='ACTUAL_ACTION_CONFORMANCE_AND_PAUSE_WINDOWS_ONLY', source_sha256=source_sha, cells=cells,
        all_checks_pass=all(all(c['checks'].values()) for c in cells.values()),
        totals={k: sum(c['counts'][k] for c in cells.values()) for k in ('calls', 'eligible', 'allocated', 'denied')},
        notes=['All six cells and all action requests retained, including denied allocations and all actual preemptions.',
            'Actions link to raw source IDs and that request preemption ordinal, not an assumed receipt-step offset.',
            'Receipt checks cover recorded eligibility fields. Status, zero computed/in-flight/spec tokens, delay_cache_blocks, encoder/lookahead guards and max_model_len are not independently recorded; the source hash binds the inspected implementation.',
            'Unchanged external/new counts are checked by source pass-through and matching measured source hash, not a downstream argument trace.',
            'No data-ready probe exists in this performance group. Pause improvement cannot establish disappearance of ready-to-schedule waiting.'])
    output = args.output or args.results / 'reservation_checks.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(output); print(json.dumps(dict(all_checks_pass=report['all_checks_pass'], totals=report['totals'])))

if __name__ == '__main__':
    main()
