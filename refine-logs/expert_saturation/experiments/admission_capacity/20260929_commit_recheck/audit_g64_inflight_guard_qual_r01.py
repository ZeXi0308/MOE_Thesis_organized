"""Read-only audit of the one corrected-adapter native qualification.

No performance ranking. Failures remain in the result with all requests counted.
Run with Python 3.10 or newer; output must be a new derived file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze_g64_liveness_diag_r02 import verify_archive

BASE = Path(__file__).resolve().parent
PACKAGE = BASE / 'candidate_g64_inflight_guard_qual_r01'
PLAN_SHA = '97a6cf841a49ce24b0392c2a65eda121b53d5262562355483fe0aa94aa05b182'
MANIFEST_SHA = '2013572450702fc7033cdfe8d35a5d781a57c9c0fbe7ec8aa986b3ec243160d6'
ADAPTER_SHA = 'ccbddc6d9860a1efc2c78dcf6614883734b2388de19a3db8034632bd9b3e9fb3'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(session):
    if sha(session / 'plan.json') != PLAN_SHA or sha(PACKAGE / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('Frozen plan/package identity mismatch')
    manifest = read(PACKAGE / 'manifest.json')
    for name, digest in manifest.items():
        if sha(PACKAGE / name) != digest:
            raise ValueError(f'Qualification package drift: {name}')
    receipt = read(session / 'receipt.json')
    if receipt['plan_sha256'] != PLAN_SHA or len(receipt['cells']) != 1:
        raise ValueError('Wrong qualification session')
    cell_receipt = receipt['cells'][0]
    cell = session / 'cell-00-ltr_t30_q1'
    archive, output_hashes, output_manifest_sha = verify_archive(cell)
    raw = read(archive / 'raw.json')
    data = read(archive / 'selective-store.json')
    observer = read(archive / 'no-progress-observer.json')
    environment = read(archive / 'environment.json')
    for name, digest in environment['source_sha256'].items():
        if digest != manifest['pkg/' + name]:
            raise ValueError(f'Executed source differs: {name}')
    if environment['source_sha256']['ltr_style_native.py'] != ADAPTER_SHA:
        raise ValueError('Wrong adapter revision')
    expected_native = {'v1/core/sched/scheduler.py': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941', 'v1/core/kv_cache_manager.py': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf', 'v1/core/block_pool.py': '202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661', 'v1/worker/gpu_model_runner.py': '81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0', 'v1/core/kv_cache_coordinator.py': '4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2', 'v1/core/single_type_kv_cache_manager.py': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8', 'v1/core/kv_cache_utils.py': '6add6f1d60634b0819833675d86be4adf00c13fe5d0a1605074353b55d3e2d39', 'config/model.py': '7d52e060db54067a21591882bd6e3c2dd2486ac9041dde1117f97023e71bf89f'}
    for name, digest in environment['vllm_source_sha256'].items():
        if expected_native[name] != digest:
            raise ValueError(f'Native runtime drift: {name}')
    workload = read(PACKAGE / 'pkg/inputs/workload.json')
    rows = raw['requests']
    expected_rows = workload['source_requests']
    if len(rows) != 64 or len({r['request_id'] for r in rows}) != 64:
        raise ValueError('Full external cohort missing or duplicated')
    for i, (row, expected) in enumerate(zip(rows, expected_rows)):
        for key in ('request_id', 'document_id', 'prompt_token_ids_sha256', 'max_output_tokens'):
            if row[key] != expected[key]:
                raise ValueError(f'Request identity drift: {i}/{key}')
        if row['arrival_s'] != workload['arrival_traces_s']['steady'][i]:
            raise ValueError('Arrival trace drift')
        if row['prompt_tokens'] != expected['prompt_token_count']:
            raise ValueError('Prompt size drift')
        if len(row['output_token_ids']) != len(row['token_times_s']):
            raise ValueError('Token/time accounting mismatch')
    first = data.get('first_direct_guard_rejection') or {}
    values = [first.get(k) for k in ('F_free_blocks', 'N_target_remaining_blocks', 'R_inflight_reserved_blocks')]
    conflict = all(type(v) is int for v in values) and values[0] >= values[1] and values[2] > 0 and values[1] + values[2] > values[0]
    host = read(archive / 'host-before.json')
    cap = read(archive / 'safe-cap-qualification.json')
    memory = read(archive / 'memory-after-init.json')
    checks = dict(
        process_complete=receipt['status'] == 'CELLS_COMPLETE' and cell_receipt['exit_code'] == 0 and not cell_receipt['timed_out'],
        archive_verified=cell_receipt['archive_status'] == 'VERIFIED',
        gpu_released=cell_receipt['gpu_process_state_after'] == 'EMPTY' and not cell_receipt['gpu_process_rows_after'],
        shared_lock=receipt['held_lock_device_inode'] == '2304:25841682495',
        all_64_completed=raw['status'] == 'COMPLETE' and raw['error'] is None and all(r['status'] == 'completed' for r in rows),
        no_no_progress_snapshot=observer['status'] == 'UNINSTALLED_WITHOUT_SNAPSHOT' and observer['snapshot'] is None,
        observer_errors_empty=not observer['observer_errors'],
        guard_rejection_observed=data.get('direct_guard_rejected_step_targets', 0) > 0,
        first_rejection_resource_conflict=conflict,
        fixed_gpu_kv=cap['status'] == 'QUALIFIED' and cap['usable_blocks'] == 4096 and memory['kv_storage_bytes'] == 8592031744,
        fixed_host_kv=host['cpu_kv']['unique_storage_bytes'] == 16 * 1024**3 and host['manager']['capacity_blocks'] == 8192,
        warmup_complete=all(read(archive / f'warmup-{i}.json')['status'] == 'COMPLETE' for i in range(3)),
    )
    return dict(status='QUALIFIED_NATIVE_GUARD' if all(checks.values()) else 'NOT_QUALIFIED',
        checks=checks, session=str(session), plan_sha256=PLAN_SHA, manifest_sha256=MANIFEST_SHA,
        adapter_sha256=ADAPTER_SHA, archive_files_verified=len(output_hashes), output_manifest_sha256=output_manifest_sha,
        requests=len(rows), completed=sum(r['status'] == 'completed' for r in rows),
        output_tokens=sum(len(r['output_token_ids']) for r in rows),
        finish_reasons=dict(Counter(r.get('finish_reason', r.get('stop_reason')) or 'unavailable' for r in rows)),
        observation_end_s=raw['observation_end_s'], group_elapsed_wall_s=receipt['elapsed_wall_s'],
        forced_rotations=data['applied_rotations'], actual_preemptions=raw['actual_preemption_count'],
        guard_checked_step_targets=data.get('direct_guard_checked_step_targets'),
        guard_rejected_step_targets=data.get('direct_guard_rejected_step_targets'),
        first_guard_rejection=first,
        scope='Native completion and exercised resource guard in this diagnostic run; not a speed comparison, tensor equality, global liveness proof or algorithm contribution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.session)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
