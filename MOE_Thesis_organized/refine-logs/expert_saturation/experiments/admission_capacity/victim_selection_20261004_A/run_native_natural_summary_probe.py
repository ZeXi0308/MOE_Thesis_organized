"""One Instruct normal-capacity profile and one natural-summary reference run.

Reuse the existing seeded controller, lock, first-boundary handoff, deadlines,
archive receipts and isolated caches. This is feasibility evidence, not a policy
comparison. No CUDA initialization happens in this entry module itself.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys

import run_native_group_seeded_base as base
import run_native_host_group_handoff as handoff
import triton_seed


MODEL_ID = 'allenai/OLMoE-1B-7B-0924-Instruct'
REVISION = '7f1c97f440f06ce36705e4f2b843edb5925f4498'
LOCK_PATH = '/root/autodl-tmp/moe-research-gpu.lock'
SOURCE_ROOT = Path('/root/autodl-tmp/moe-a-victim-20261004')
OUTPUT_ROOT = Path('/root/moe-a-victim-20261007')
LABELS = ('profile', 'remaining-budget')


def verify_model(plan, config, env):
    """Same exact offline checks as the base verifier, fixed to this Instruct."""
    model = config['model']
    base.require(model['id'] == MODEL_ID and
                 model['revision'] == model['tokenizer_revision'] == REVISION and
                 model['dtype'] == 'bfloat16', 'Wrong Instruct model identity')
    hub = Path(plan['hf_cache_dir']) / 'hub'
    snapshot = hub / 'models--allenai--OLMoE-1B-7B-0924-Instruct' / 'snapshots' / REVISION
    base.require(snapshot.is_dir(), 'Private pinned Instruct snapshot missing')
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    base.require(len(names) == 3, 'Expected three existing Instruct weight shards')
    expected_files = plan['model_files']
    required = set(names) | {'config.json', 'model.safetensors.index.json',
                             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'}
    base.require(required <= set(expected_files), 'Incomplete frozen model/tokenizer hashes')
    files = {}
    for name, expected in expected_files.items():
        base.require(Path(name).name == name, 'Invalid model file name')
        path = snapshot / name
        resolved = path.resolve(strict=True)
        base.require(resolved.is_relative_to(hub.resolve()) and resolved.is_file()
                     and resolved.stat().st_size > 0, f'Model file absent/outside private cache: {name}')
        sha = base.digest(resolved)
        base.require(sha == expected['sha256'], f'Pinned model identity differs: {name}')
        if expected['bytes'] is not None:
            base.require(resolved.stat().st_size == expected['bytes'], f'Model file size differs: {name}')
        if path.is_symlink() and len(resolved.name) == 64:
            base.require(sha == resolved.name, f'HF content-addressed blob mismatch: {name}')
        if name in config['source']['tokenizer_files_sha256']:
            base.require(sha == config['source']['tokenizer_files_sha256'][name],
                         f'Tokenizer identity differs: {name}')
        files[name] = dict(bytes=resolved.stat().st_size, sha256=sha)
    check = ('import json; from huggingface_hub import snapshot_download; '
             f'print(json.dumps(snapshot_download({MODEL_ID!r}, '
             f'revision={REVISION!r}, local_files_only=True)))')
    resolved = json.loads(base.command([plan['python'], '-c', check], env=env, timeout=60))
    base.require(Path(resolved).resolve() == snapshot.resolve(), 'Offline HF resolution uses another snapshot')
    return dict(status='PRIVATE_PINNED_SNAPSHOT_COMPLETE_OFFLINE', model_id=MODEL_ID,
                revision=REVISION, snapshot=str(snapshot), files=files,
                scope='Exact frozen file hashes, tokenizer hashes, three weight-index shards and offline HF resolution')


def configure(plan):
    base.require(plan['kind'] == 'PRO6000_NATIVE_NATURAL_SUMMARY_PROBE' and
                 plan['experiment_role'] == 'NATURAL_SUMMARY_FEASIBILITY', 'Wrong experiment role')
    base.require(re.fullmatch(r'[0-9a-f]{64}', plan.get('package_manifest_sha256', '')) is not None,
                 'Package is not frozen')
    base.require(plan['lock_path'] == LOCK_PATH, 'Must use the existing shared lock')
    base.require(Path(plan['source_package']) == SOURCE_ROOT / 'candidate_native_natural_summary_probe_r01',
                 'Wrong isolated source package')
    base.require(Path(plan['runtime_overlay']) == SOURCE_ROOT / 'runtime-overlay', 'Wrong runtime overlay')
    base.require(Path(plan['session_dir']).parent == OUTPUT_ROOT, 'Wrong authorized output root')
    base.require(plan['model_revision'] == REVISION, 'Wrong model revision in plan')
    base.require(plan['approved_total_wall_seconds'] == 1200 and
                 type(plan['per_cell_wall_seconds']) is int and
                 0 < plan['per_cell_wall_seconds'] <= 600, 'Require bounded 1200-second group and <=600-second cells')
    config = plan['configuration']
    base.require(config['max_num_seqs'] == 384 and config['max_seconds'] == 600,
                 'Keep the existing sequence cap and measurement horizon')
    cells = plan['cells']
    base.require(plan['arms'] == list(LABELS) and [c['label'] for c in cells] == list(LABELS),
                 'Exactly one profile followed by one reference measurement is allowed')
    base.require([c['profile_only'] for c in cells] == [True, False] and
                 [c['requests'] for c in cells] == [0, 320] and
                 all(c.get('input_case') is None and c['native_victim_rule'] == 'remaining_budget'
                     and c.get('funding_victim_rule', 'tail') == 'tail'
                     and c.get('victim_rule', 'tail') == 'tail' for c in cells),
                 'Cells must use root natural inputs and the same remaining-budget rule')
    base.require(plan.get('pinned_gpu_kv_bytes') is None, 'Old-model KV capacity cannot seed this profile')
    base.require(plan['runtime_cache_initial'] == 'TRITON_SEEDED_PRIVATE_OTHER_CACHES_EMPTY',
                 'Explicit identical private-cache initialization required')
    seed = Path(plan['triton_cache_seed']['path'])
    expected = plan['triton_cache_seed']['normalized_sha256']
    base.require(seed.parent == OUTPUT_ROOT and seed.is_dir(), 'Seed must be in A output root')
    base.require(triton_seed.inspect_seed(seed)['normalized_sha256'] == expected, 'Triton seed differs')
    base.MANIFEST = plan['package_manifest_sha256']
    base.REVISION = REVISION
    base.ARMS = LABELS
    base.verify_model = verify_model
    original_env = base.private_env

    def env(p, cache, arm, memory_file):
        result = original_env(p, cache, 'queue_fund', memory_file)
        result.update(A_NATIVE_OLDEST_ADMISSION='queue_fund', A_NATIVE_OLDEST_REPEAT='1',
                      A_NATIVE_VICTIM_RULE='remaining_budget', A_NATIVE_VICTIM_FULL_RUNNING='off',
                      A_NATIVE_VICTIM_CURRENT_GUARD='off', A_SELF_PREEMPT_CONTINUE='off',
                      A_NATIVE_CAPACITY_DEFERRAL='off', A_RECOVERY_LEASE_MODE='q1',
                      A_FUNDING_VICTIM_RULE='tail', A_PROFILE_ONLY='0',
                      A_MAX_NUM_SEQS='384', A_MAX_SECONDS='600', PYTHONPATH=p['runtime_overlay'])
        # No A_INPUT_CASE: run.sh's --inputs inputs uses the root natural corpus.
        if arm == 'ordinary':
            return result
        base.require(arm in LABELS, 'Unexpected cell')
        record = triton_seed.install(seed, cache / 'triton', expected)
        record.update(source=str(seed), rule=result['A_NATIVE_VICTIM_RULE'],
                      cache_initial=p['runtime_cache_initial'])
        base.write_json(cache.parent / 'triton-cache-seed.json', record, new=True)
        if arm == 'profile':
            result['A_PROFILE_ONLY'] = '1'
        else:
            profile_dir = Path(p['session_dir']) / 'cell-00-profile' / 'output'
            profile = json.loads((profile_dir / 'normal_capacity_profile.json').read_text())
            engine = json.loads((profile_dir / 'engine_args.json').read_text())
            base.require(profile['status'] == 'PROFILE_COMPLETE' and
                         engine['model'] == MODEL_ID and engine['revision'] == REVISION and
                         engine['tokenizer_revision'] == REVISION and engine['dtype'] == 'bfloat16' and
                         engine['gpu_memory_utilization'] == .90 and
                         'kv_cache_memory_bytes' not in engine, 'This Instruct normal .90 profile is required')
            capacity = profile['pin_kv_cache_memory_bytes']
            base.require(type(capacity) is int and capacity > 0, 'Invalid profiled KV capacity')
            result['A_GPU_KV_BYTES'] = str(capacity)
        return result

    base.private_env = env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    configure(plan)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('SIGTERM')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Group budget')))
    return base.run(plan, args.plan)


if __name__ == '__main__':
    handoff.group = sys.modules[__name__]
    print(f'NATURAL_SUMMARY_PROBE_ENTRY controller_pid={os.getpid()} '
          f'entry_sha256={base.digest(__file__)}', flush=True)
    raise SystemExit(handoff.main())
