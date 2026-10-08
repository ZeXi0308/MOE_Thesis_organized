#!/usr/bin/env python3
"""Frozen retry metrics plus persistent-gate pass-through and terminal observations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '4bb4157ebf3b12686336aaa736aec25cebb2b480927f9204979a974598178953'
EXPECTED_POLICY = dict(
    source_sha256='4eae969ca391e01d8ab44c516115f9fdc702e12537a01240576c0484b98579e8',
    name='persistent_target_gate',
    frozen_source_sha256='81e1e3d1a1a4d13336188242876c6c29ab2e298da62c9ef3735af7104c0917f0',
    compiled_source_sha256='3243b5052bef0ea2e5a97d1e8593284f858d19a8211b9932bc7f78d0202346dd',
    config_revision='persistent_active_head_passthrough')


def load_parent():
    path = ROOT/'analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen retry-defer analyzer changed')
    spec = importlib.util.spec_from_file_location('persistent_retry_parent_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def terminal(row, number):
    reason = row.get('release_reason'); release = row.get('release_s'); deadline = row.get('deadline_s')
    if not row.get('actual_gate_executed'): category = 'NO_EXECUTED_GATE'
    elif reason is None: category = 'ACTIVE_OR_RELEASE_UNOBSERVED'
    elif reason == 'DEADLINE': category = 'DEADLINE_WINDOW_COMPLETED'
    elif reason == 'COHORT_FINISHED_AND_FREED': category = 'COHORT_COMPLETION_RELEASE'
    elif reason == 'RUNNING_EMPTY': category = 'RUNNING_EMPTY_RELEASE'
    elif reason == 'UNINSTALL_WITH_ACTIVE_GATE': category = 'OBSERVATION_ENDED_WITH_ACTIVE_GATE'
    elif reason in ('BASELINE_OVERRIDE', 'NEW_WAITER', 'TARGET_LEFT_PREEMPTED_QUEUE',
                    'TARGET_NATIVE_STATE_CHANGED', 'NATIVE_HEAD_OR_QUEUE_CHANGED'):
        category = 'SAFETY_OR_INTERFACE_ABORT'
    else: category = 'UNKNOWN_RELEASE_REASON'
    return dict(classification=category, recorded_release_reason=reason,
        deadline_window_completed=category == 'DEADLINE_WINDOW_COMPLETED' and
            number(release) and number(deadline) and release >= deadline,
        observed_lifetime_reached_deadline=release >= deadline
            if row.get('actual_gate_executed') and number(release) and number(deadline) else None,
        observed_gate_lifetime_s=row.get('selection_to_release_s'),
        first_executed_break_to_release_s=row.get('first_break_to_release_s'),
        timing_semantics='Host-observed gate lifetime, including overlapping native pass-through and '
            'periods when the native decision boundary is not reached. Not counterfactual added latency '
            'or saved latency; release removes only the extra gate.')


def extend_actions(result, data, raw, cell, number):
    result['native_queue_pass_through_count'] = 0
    result['native_queue_pass_through_records'] = []
    result['terminal_classification_counts'] = {}
    result['recorded_interface_adapter'] = (data or {}).get('interface_adapter')
    if data is None or raw is None: return result
    failures = list(result.get('failed_checks', [])); missing = list(result.get('missing', []))
    adapter = data.get('interface_adapter')
    if not isinstance(adapter, dict): missing.append('interface_adapter')
    else:
        for key in ('name', 'frozen_source_sha256', 'compiled_source_sha256'):
            if key not in adapter: missing.append('interface_adapter.'+key)
            elif adapter[key] != EXPECTED_POLICY[key]: failures.append('interface_adapter_mismatch.'+key)
    origin = raw['measurement_origin_perf_counter_s']; mapping = raw.get('internal_to_source', {})
    rows = result.get('rows', [])
    events = data.get('events') if isinstance(data.get('events'), list) else []
    for index, event in enumerate(events):
        if event.get('kind') != 'native_queue_pass_through': continue
        absent = [key for key in ('host_perf_s', 'target', 'actual_head', 'actual_queue', 'skipped_order') if key not in event]
        missing.extend('pass_through.'+key for key in absent)
        stamp = event.get('host_perf_s'); target = event.get('target'); head = event.get('actual_head')
        linked = [row for row in rows if row['target']['internal_request'] == target]
        row = linked[0] if len(linked) == 1 else None
        if cell['mode'] != 'defer_once': failures.append('native_has_active_gate_pass_through')
        if target is not None and head == target: failures.append('pass_through_head_equals_target')
        if not number(stamp): missing.append('pass_through.valid_host_clock')
        if row is None: missing.append('pass_through.selected_target_link')
        elif not row['actual_gate_executed']: failures.append('pass_through_without_executed_gate')
        relative = stamp-origin if number(stamp) else None
        if row is not None and number(relative):
            first, release = row.get('first_break_s'), row.get('release_s')
            if number(first) and relative < first: failures.append('pass_through_precedes_first_break')
            if number(release) and relative > release: failures.append('pass_through_after_release')
        result['native_queue_pass_through_records'].append(dict(event_index=index, host_s=relative,
            target=dict(internal_request=target, source_request=mapping.get(target)),
            actual_head=dict(internal_request=head, source_request=mapping.get(head)),
            actual_queue=event.get('actual_queue'), skipped_order=event.get('skipped_order'),
            linked_decision_s=row.get('decision_s') if row else None,
            since_decision_s=relative-row['decision_s'] if row and number(relative) else None,
            until_release_s=row['release_s']-relative if row and number(row.get('release_s')) and number(relative) else None,
            raw_record=event))
    passes = result['native_queue_pass_through_records']; result['native_queue_pass_through_count'] = len(passes)
    for row in rows:
        row['native_queue_pass_through_records'] = [record for record in passes
            if record['target']['internal_request'] == row['target']['internal_request']]
        row['native_queue_pass_through_count'] = len(row['native_queue_pass_through_records'])
        row['gate_terminal_observation'] = terminal(row, number)
        classification = row['gate_terminal_observation']['classification']
        result['terminal_classification_counts'][classification] = result['terminal_classification_counts'].get(classification, 0)+1
        if row['release_reason'] == 'DEADLINE' and not row['gate_terminal_observation']['deadline_window_completed']:
            failures.append('deadline_release_before_declared_deadline')
    result.update(failed_checks=sorted(set(failures)), missing=sorted(set(missing)),
        status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED')
    return result


def analyze_session(session):
    parent = load_parent(); original_actions = parent.actions
    parent.actions = lambda data, raw, cell, allocations, request_evidence: extend_actions(
        original_actions(data, raw, cell, allocations, request_evidence), data, raw, cell, parent.number)
    result = parent.analyze_session(session)
    for cell in result['cells']:
        config_path = Path(cell['directory'])/'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        actions = cell['retry_defer_actions']; recorded = actions.get('recorded_interface_adapter')
        revision = config.get('recovery_retry_defer_revision')
        policy_sha = config.get('recovery_retry_defer_policy_sha256')
        evidence = dict(expected_policy=EXPECTED_POLICY, recorded_interface_adapter=recorded,
            recorded_config_revision=revision, recorded_policy_sha256=policy_sha,
            recorded_runner_sha256=config.get('recovery_retry_defer_persistent_runner_sha256'))
        evidence['version_verified'] = isinstance(recorded, dict) and all(
            recorded.get(key) == EXPECTED_POLICY[key] for key in ('name', 'frozen_source_sha256', 'compiled_source_sha256')) and revision == EXPECTED_POLICY['config_revision'] and policy_sha == EXPECTED_POLICY['source_sha256']
        cell['persistent_retry_policy_evidence'] = evidence
        for name, recorded_value, expected_value in (
            ('recovery_retry_defer_revision', revision, EXPECTED_POLICY['config_revision']),
            ('recovery_retry_defer_policy_sha256', policy_sha, EXPECTED_POLICY['source_sha256'])):
            if recorded_value != expected_value:
                key = 'missing' if recorded_value is None else 'failed_checks'
                actions.setdefault(key, []).append('config.'+name)
                if actions['status'] != 'UNAVAILABLE':
                    actions['status'] = 'FAIL' if actions.get('failed_checks') else 'UNVERIFIED'
    by_path = {cell['directory']: cell for cell in result['cells']}
    for pair in result['comparisons']:
        pair['persistent_retry_observations'] = {key: {name: by_path[pair[key]]['retry_defer_actions'].get(name)
            for name in ('status', 'actual_gate_count', 'requested_break_count', 'executed_break_count',
                         'native_queue_pass_through_count', 'terminal_classification_counts')}
            if pair.get(key) in by_path else None for key in ('candidate', 'native')}
        for key in ('candidate', 'native'):
            if pair.get(key) in by_path:
                pair['retry_execution_observations'][key]['status'] = by_path[pair[key]]['retry_defer_actions']['status']
    result['execution_layout']['design'] = 'ONE_TARGET_PERSISTENT_RETRY_DEFERRAL'
    result['persistent_retry_semantics'] = (
        'Persistent gate v2 allows a different native head through the callback while retaining the selected '
        'target gate. A pass-through records entry to the original native handler, not successful allocation '
        'or model execution. Requested and executed target breaks remain distinct, and one gate can break '
        'several scheduler iterations. Release classification and recorded reason distinguish deadline '
        'completion, cohort/running-empty release, safety/interface abort and observation ending. Observed '
        'gate lifetime includes native gate-closed periods and overlapping peer progress; it is not measured '
        'counterfactual added latency or saved latency. All-request statistics, actual target allocations '
        'during the gate, costs, storage/GC accounting, zero-action and early-stop handling, and original '
        'nearest-native pair calculations are inherited unchanged. No synthetic unstarted arms are added.')
    result['analyzer_sources_sha256']['recovery_retry_defer/analyze_persistent.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'], cells=[dict(
        directory=cell['directory'], **{key: cell['retry_defer_actions'].get(key) for key in
        ('status', 'actual_gate_count', 'executed_break_count', 'native_queue_pass_through_count',
         'terminal_classification_counts')}) for cell in result['cells']])))


if __name__ == '__main__':
    main()
