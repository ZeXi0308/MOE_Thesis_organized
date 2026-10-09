#!/usr/bin/env python3
"""Shared-prefix tail characterization with a fresh same-group capacity profile."""
import argparse
import csv
import json
from pathlib import Path

import analyze_native_shared_prefix_probe as shared


base, mixed = shared.base, shared.mixed
CAPACITY_FIELDS = ('block_size_tokens', 'total_blocks', 'usable_blocks', 'null_blocks',
    'kv_page_bytes', 'actual_kv_storage_bytes', 'pin_kv_cache_memory_bytes',
    'host_kv_bytes', 'host_capacity_blocks', 'max_num_seqs', 'max_model_len',
    'max_num_batched_tokens', 'gpu_memory_utilization')


def validate_cells(plan):
    expected = [dict(label='profile', profile_only=True, requests=0),
                dict(label='shared-prefix-tail', profile_only=False, requests=320)]
    cells = plan.get('cells', [])
    if (plan.get('kind') != 'PRO6000_NATIVE_EQUAL_HELD_ONCE'
            or plan.get('experiment_role') != shared.ROLE or len(cells) != 2
            or 'pinned_gpu_kv_bytes' in plan):
        raise ValueError('Expected a fresh-profile shared-prefix plan without a prior capacity pin')
    for spec, required in zip(cells, expected):
        required.update(input_case='high', native_victim_rule='tail', funding_victim_rule='tail')
        if any(spec.get(k) != value or type(spec.get(k)) is not type(value)
               for k, value in required.items()):
            raise ValueError('Expected profile cell00 followed by 320-request shared-prefix-tail cell01')
    if plan.get('arms') != [c['label'] for c in cells]:
        raise ValueError('Plan arm order differs from profile/measurement cells')
    return cells


def gpu_uuid(environment):
    device = environment.get('gpu_before', {}).get('device')
    if not isinstance(device, str):
        return None
    rows = list(csv.reader(device.splitlines()))
    return rows[0][1].strip() if len(rows) == 1 and len(rows[0]) >= 2 else None


def read_profile(directory):
    archive = base.archive_dir(directory)
    return dict(directory=str(directory), capacity=base.optional(archive / 'normal_capacity_profile.json', {}),
        engine_status=base.optional(archive / 'status.json', {}),
        config=base.optional(archive / 'config.json', {}), engine_args=base.optional(archive / 'engine_args.json', {}),
        environment=base.optional(archive / 'environment.json', {}),
        timing=base.optional(archive / 'timing.json', {}))


def capacity_check(profile, measured, environment, plan):
    capacity, actual = profile['capacity'], measured.get('capacity_profile', {})
    engine, measured_engine = profile['engine_args'], measured.get('engine_args', {})
    status, config = profile['engine_status'], profile['config']
    wanted_uuid = plan.get('authorized_gpu_uuid')
    positive = lambda value: type(value) is int and value > 0
    pin = capacity.get('pin_kv_cache_memory_bytes')
    checks = dict(profile_status=status.get('status') == 'PROFILE_COMPLETE' and status.get('error') is None,
        profile_has_no_measurement=status.get('requests_completed') == 0
            and status.get('measurement_status') == 'UNRUN' and status.get('warmup_status') == 'UNRUN',
        capacity_profile_complete=capacity.get('status') == 'PROFILE_COMPLETE'
            and capacity.get('profile_only') is True,
        normal_profile_without_pin=config.get('profile_only') is True
            and config.get('fixed_kv_cache_memory_bytes', 'MISSING') is None
            and engine.get('kv_cache_memory_bytes') is None,
        profile_prefix_enabled=config.get('enable_prefix_caching') is True
            and engine.get('enable_prefix_caching') is True,
        positive_profile_pin=positive(pin),
        measurement_pin_verified=actual.get('status') == 'PIN_VERIFIED' and actual.get('profile_only') is False,
        measurement_engine_uses_profile_pin=positive(pin) and measured_engine.get('kv_cache_memory_bytes') == pin,
        measurement_config_uses_profile_pin=positive(pin)
            and (measured.get('config') or {}).get('fixed_kv_cache_memory_bytes') == pin,
        profile_gpu_uuid=isinstance(wanted_uuid, str) and bool(wanted_uuid) and gpu_uuid(profile['environment']) == wanted_uuid,
        measurement_gpu_uuid=isinstance(wanted_uuid, str) and bool(wanted_uuid) and gpu_uuid(environment) == wanted_uuid,
        same_engine_configuration=bool(engine) and {k: v for k, v in engine.items() if k != 'kv_cache_memory_bytes'}
            == {k: v for k, v in measured_engine.items() if k != 'kv_cache_memory_bytes'},
        same_candidate_source=bool(profile['environment'].get('source_sha256'))
            and profile['environment']['source_sha256'] == environment.get('source_sha256'),
        same_native_source=bool(profile['environment'].get('vllm_source_sha256'))
            and profile['environment']['vllm_source_sha256'] == environment.get('vllm_source_sha256'))
    for key in CAPACITY_FIELDS:
        checks['matching_' + key] = key in capacity and key in actual and capacity[key] == actual[key]
    for label, row in (('profile', capacity), ('measurement', actual)):
        valid = all(positive(row.get(k)) for k in CAPACITY_FIELDS if k not in ('null_blocks', 'gpu_memory_utilization'))
        valid = valid and base.count_known(row.get('null_blocks'))
        checks[label + '_physical_capacity_accounting'] = valid and (
            row['total_blocks'] == row['usable_blocks'] + row['null_blocks']
            and row['actual_kv_storage_bytes'] == row['pin_kv_cache_memory_bytes']
            == row['total_blocks'] * row['kv_page_bytes']
            and row['host_kv_bytes'] == row['host_capacity_blocks'] * row['kv_page_bytes'])
    return dict(status='VERIFIED' if all(checks.values()) else 'MISMATCH_OR_MISSING', checks=checks,
        authorized_gpu_uuid=wanted_uuid, profile_gpu_uuid=gpu_uuid(profile['environment']),
        measurement_gpu_uuid=gpu_uuid(environment), profile_pin_kv_cache_memory_bytes=pin,
        semantics='Capacity comes only from cell00 in this session. The measurement must use that exact GPU allocation and matching physical Host/GPU page layout on the authorized UUID. Missing or failed profiling cannot inherit an old device capacity.')


def analyze(session, policy_source, input_config):
    plan = base.read(session / 'plan.json')
    _, spec = validate_cells(plan)
    profile = read_profile(session / 'cell-00-profile')
    pin = profile['capacity'].get('pin_kv_cache_memory_bytes')
    # Supply the newly observed pin to the inherited checker, without changing
    # the frozen on-disk plan or recording it as an a priori configuration.
    check_plan = dict(plan)
    if type(pin) is int and pin > 0:
        check_plan['pinned_gpu_kv_bytes'] = pin
    directory = session / 'cell-01-shared-prefix-tail'
    cell, raw = base.read_native_cell(directory, spec, check_plan,
        plan.get('configuration', {}).get('max_seconds', 600), policy_source)
    archive = base.archive_dir(directory)
    store = base.optional(archive / 'selective-store.json', {})
    environment = base.optional(archive / 'environment.json', {})
    executed = directory / 'candidate_native_oldest_strong_r01/pkg'
    source = executed / 'staged_store_rotation.py'
    source = source if source.is_file() else policy_source
    config_path = executed / 'inputs/pro_high/config.json'
    executed_input = config_path.is_file()
    config_path = config_path if executed_input else input_config
    budget = inputs = dict(status='INPUT_UNAVAILABLE')
    if config_path.is_file() and config_path.with_name('workload.json').is_file():
        config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
        budget = mixed.budget_check(config, workload, cell.get('config') or {}, raw)
        inputs = shared.input_check(config, workload, raw)
        inputs.update(input_config_path=str(config_path), input_config_sha256=mixed.sha(config_path),
            input_workload_sha256=mixed.sha(config_path.with_name('workload.json')),
            input_source='EXECUTED_CELL_PACKAGE' if executed_input else 'SUPPLIED_PACKAGE_FALLBACK')
    warmup = mixed.warmup_check(archive, cell.get('engine_args', {}))
    domain = shared.domain_check(cell, store)
    observations = shared.shared_observations(cell, store, raw, source,
        environment.get('source_sha256', {}).get('staged_store_rotation.py'))
    capacity = capacity_check(profile, cell, environment, plan)
    complete = cell.get('status') == 'COMPLETE' and all(value.get('status') == 'VERIFIED'
        for value in (budget, inputs, warmup, domain, observations, capacity))
    return dict(status='COMPLETE_CHARACTERIZATION' if complete else
        'NO_MEASUREMENT' if raw is None else 'INCOMPLETE_OR_INVALID', experiment_role=shared.ROLE,
        session=str(session), plan=plan, receipt=base.optional(session / 'receipt.json'),
        profile_cell=profile, profile_capacity_check=capacity, cell=cell,
        budget_consistency=budget, input_consistency=inputs, fixed_warmup_check=warmup,
        shared_prefix_domain=domain, shared_prefix_observations=observations,
        analysis_code_sha256={Path(module.__file__).name: mixed.sha(Path(module.__file__)) for module in (base, mixed, shared)}
            | {Path(__file__).name: mixed.sha(Path(__file__))}, semantics=shared.SEMANTICS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--policy-source', type=Path, default=shared.PACKAGE / 'pkg/staged_store_rotation.py')
    parser.add_argument('--input-config', type=Path, default=shared.PACKAGE / 'pkg/inputs/pro_high/config.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.session.resolve(), args.policy_source.resolve(), args.input_config.resolve())
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output)), ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
