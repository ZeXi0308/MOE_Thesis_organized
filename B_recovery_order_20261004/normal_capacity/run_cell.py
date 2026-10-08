#!/usr/bin/env python3
"""Thin normal-capacity adapter over the immutable B cell runner.

Only workload bounds, KV allocation, fixed concurrency, and timeout differ.
The original warmup, native policy, transfer observer, execution and teardown
remain in the pinned parent runner. Every replacement must match exactly once.
"""
import hashlib
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'pkg'
PARENT_SHA256 = '7046411c3b19a4d023004ea873f807aac0719aa882246649e671ea9e5d073be9'
ROTATION_SHA256 = '11b986fe0a28fc8931e6785229935cd98027cbe8475a3d61c2ac792f29c428cb'


def gpu_kv_argument(value):
    if value == 'auto':
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError('--gpu-kv-bytes must be auto or positive bytes') from error
    if result <= 0:
        raise ValueError('--gpu-kv-bytes must be positive')
    return result


def validate_args(args):
    if not 1 <= args.max_num_seqs <= 1024:
        raise ValueError('max-num-seqs must be within the fixed 1024-token batch budget')
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError('max-seconds must be positive and finite')
    for key, expected in {'A_NATIVE_OLDEST_ADMISSION': 'native',
            'A_NATIVE_OLDEST_REPEAT': '1', 'A_RECOVERY_LEASE_MODE': 'off',
            'A_NATIVE_VICTIM_RULE': 'tail', 'A_NATIVE_VICTIM_FULL_RUNNING': 'off',
            'A_NATIVE_CAPACITY_DEFERRAL': 'off', 'A_NATIVE_VICTIM_CURRENT_GUARD': 'off',
            'A_SELF_PREEMPT_CONTINUE': 'off'}.items():
        if os.environ.setdefault(key, expected) != expected:
            raise ValueError(f'Normal-capacity cells require {key}={expected}')
    os.environ.setdefault('B_RECOVERY_ORDER', 'native')


def validate_workload(config, workload, args):
    global FIXED_CAP
    expected_model = dict(id='allenai/OLMoE-1B-7B-0924',
        revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5',
        tokenizer_revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5', dtype='bfloat16')
    if any(config['model'].get(key) != value for key, value in expected_model.items()):
        raise ValueError('Normal-capacity cells retain the pinned OLMoE BF16 model and tokenizer')
    rows, prompts = workload['source_requests'], workload['actual_prompt_token_ids']
    arrivals = workload['arrival_traces_s']['steady']
    if not rows or len(rows) != len(prompts) or len(rows) != len(arrivals):
        raise ValueError('Request, prompt and arrival counts must agree and be nonempty')
    if config.get('requests') != len(rows):
        raise ValueError('Signed config request count differs from workload')
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in arrivals):
        raise ValueError('Arrival times must be finite and nonnegative')
    count, cap = config['output_tokens'], config.get('cap', args.max_num_seqs)
    if type(count) is not int or count < 1 or type(cap) is not int or not 1 <= cap <= args.max_num_seqs:
        raise ValueError('Invalid output limit or fixed concurrency cap')
    overrides = config.get('output_tokens_by_request', {})
    if set(overrides) - {row['request_id'] for row in rows}:
        raise ValueError('Unknown output-limit request ID')
    maximum = 0
    for row, tokens in zip(rows, prompts):
        limit = overrides.get(row['request_id'], count)
        if type(limit) is not int or limit < 1 or not 1 <= len(tokens) < 4096 or len(tokens) + limit > 4096:
            raise ValueError('Each prompt plus requested output must fit max_model_len=4096')
        maximum = max(maximum, len(tokens) + limit)
    config['cap'] = cap
    FIXED_CAP = cap
    config['normal_capacity_max_request_tokens'] = maximum
    config['requested_gpu_kv_bytes'] = args.gpu_kv_bytes
    config['normal_capacity_adapter'] = dict(parent_sha256=PARENT_SHA256,
        staged_rotation_parent_sha256=ROTATION_SHA256,
        staged_rotation_compiled_sha256=hashlib.sha256(adapted_rotation_source().encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        safe_static_sha256=hashlib.sha256((HERE/'safe_static.py').read_bytes()).hexdigest(),
        compiled_runner_sha256=COMPILED_SHA256,
        workload_scope='All signed input requests and external arrival times retained unchanged')


def adapted_rotation_source():
    source = (PKG/'staged_store_rotation.py').read_text()
    if hashlib.sha256(source.encode()).hexdigest() != ROTATION_SHA256:
        raise RuntimeError('Parent native observation wrapper changed')
    old = "    if open_population and scheduler.max_num_running_reqs!=32:\n        raise ValueError('Open population requires unchanged running cap32')"
    if source.count(old) != 1:
        raise RuntimeError('Expected one fixed-cap check in the parent native wrapper')
    return source.replace(old, "    if open_population and scheduler.max_num_running_reqs!=NORMAL_FIXED_CAP:\n        raise ValueError('Open population cap differs from signed normal-capacity config')")


def normal_install_rotation(*args, **kwargs):
    namespace = dict(__name__='normal_capacity_native_observation',
        __file__=str(PKG/'staged_store_rotation.py'), NORMAL_FIXED_CAP=FIXED_CAP)
    exec(compile(adapted_rotation_source(), str(PKG/'staged_store_rotation.py')+'[normal_capacity]', 'exec'), namespace)
    return namespace['install'](*args, **kwargs)


def qualify_memory(engine, config, memory, qualification, args):
    if qualification['status'] != 'QUALIFIED' or not qualification['observed_scheduler_reserve_full_isl']:
        raise RuntimeError('Actual KV pool/spec or full-history reservation is unqualified')
    layout = engine.engine_core.engine_core.scheduler.kv_cache_config
    # The retained qualifier requires one plain FullAttention group, no sharing,
    # no lookahead, no speculative decoding, and a drained pool with one null block.
    group = layout.kv_cache_groups[0]
    bytes_per_block = int(group.kv_cache_spec.page_size_bytes) * len(group.layer_names)
    total_blocks = qualification['total_blocks']
    actual_bytes = memory['kv_storage_bytes']
    if bytes_per_block <= 0 or total_blocks * bytes_per_block != actual_bytes:
        raise RuntimeError('Actual unique GPU KV storage differs from pool blocks times spec bytes/block')
    if args.gpu_kv_bytes is not None and actual_bytes != args.gpu_kv_bytes:
        raise RuntimeError('Actual GPU KV storage differs from the fixed group budget; use block-aligned bytes')
    config.update(fixed_kv_cache_memory_bytes=actual_bytes,
        actual_gpu_kv_blocks=total_blocks, actual_gpu_kv_usable_blocks=qualification['usable_blocks'],
        actual_gpu_kv_bytes_per_block=bytes_per_block)
    qualification.update(actual_gpu_kv_storage_bytes=actual_bytes,
        spec_bytes_per_gpu_block=bytes_per_block, gpu_memory_utilization=.90,
        requested_gpu_kv_bytes=args.gpu_kv_bytes,
        budget_semantics='auto profiles once; use fixed_kv_cache_memory_bytes for every subsequent comparison cell')


def adapted_source():
    source = (PKG/'run_recovery_cadence.py').read_text()
    if hashlib.sha256(source.encode()).hexdigest() != PARENT_SHA256:
        raise RuntimeError('Parent runner changed; adapter needs an explicit source update')

    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError(f'Expected {count} exact adapter match(es): {old[:90]!r}')
        source = source.replace(old, new)

    replace('Single serial H1 candidate cell on the frozen 128-article holdout input.',
            'Normal-capacity native cell over signed inputs; fixed model, host KV and transfer implementation.')
    replace("    args = parser.parse_args()", "    parser.add_argument('--max-num-seqs', type=int, required=True)\n    parser.add_argument('--gpu-kv-bytes', type=gpu_kv_argument, default=None, metavar='auto|BYTES')\n    parser.add_argument('--max-seconds', type=float, default=300)\n    args = parser.parse_args()\n    validate_args(args)")
    replace("        from safe_static import qualify_safe_cap", "        from normal_capacity_safe_static import qualify_safe_cap")
    replace("        from staged_store_rotation import install as install_rotation", "        install_rotation = normal_install_rotation")
    replace("        if len(lengths) != 128 or not all(256 <= n <= 3072 for n in lengths) or workload['arrival_traces_s']['steady'] != [i * .2 for i in range(128)]:\n            raise ValueError('requires the frozen128 complete holdout prompts and steady0.2s arrivals')", "        validate_workload(config, workload, args)")
    replace("        config.update(requests=128, prompt_tokens=max(lengths), output_tokens=1024, output_mode='eos',\n            output_tokens_by_request={}, cap=32, engine_max_num_seqs=32, policy='static', max_seconds=180,", "        config.update(requests=len(lengths), prompt_tokens=max(lengths), output_tokens=config['output_tokens'], output_mode='eos',\n            output_tokens_by_request=config.get('output_tokens_by_request', {}), cap=config['cap'], engine_max_num_seqs=args.max_num_seqs, policy='static', max_seconds=args.max_seconds,")
    replace('fixed_kv_cache_memory_bytes=KV_BYTES', 'fixed_kv_cache_memory_bytes=args.gpu_kv_bytes')
    replace('max_model_len=4096, max_num_seqs=32', 'max_model_len=4096, max_num_seqs=args.max_num_seqs')
    replace('kv_cache_memory_bytes=KV_BYTES', 'kv_cache_memory_bytes=args.gpu_kv_bytes')
    replace("        if (qualification['status'] != 'QUALIFIED' or qualification['usable_blocks'] != 4096\n                or not qualification['observed_scheduler_reserve_full_isl'] or memory['kv_storage_bytes'] != KV_BYTES):\n            raise RuntimeError('actual KV pool/storage or initial-history reservation differs from fixed budget')", "        qualify_memory(engine, config, memory, qualification, args)\n        dump(out/'safe-cap-qualification.json', qualification)\n        dump(out/'config.json', config)")
    replace("            wc, ww = warmups[domain]\n            work = dict(ww, source_requests=ww['source_requests'][:count],", "            wc, ww = warmups[domain]\n            count = min(count, args.max_num_seqs, len(ww['source_requests']))\n            cap = min(cap, args.max_num_seqs)\n            if count < 1:\n                raise ValueError('Warmup domain is empty')\n            work = dict(ww, source_requests=ww['source_requests'][:count],")
    replace('set_empty_admission_cap(engine, 32)  # Verifies drain before installing either adapter arm.', "set_empty_admission_cap(engine, config['cap'])  # Same compiled engine; fixed workload concurrency.")
    replace("run_id='measured', max_seconds=180", "run_id='measured', max_seconds=args.max_seconds", count=2)
    replace("if __name__ == '__main__':\n    main()", '')
    return source


def main():
    import importlib.util
    sys.path.insert(0, str(PKG))
    spec = importlib.util.spec_from_file_location('normal_capacity_safe_static', HERE/'safe_static.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['normal_capacity_safe_static'] = module
    source = adapted_source()
    global COMPILED_SHA256
    COMPILED_SHA256 = hashlib.sha256(source.encode()).hexdigest()
    namespace = dict(__name__='normal_capacity_cell', __file__=str(PKG/'run_recovery_cadence.py'),
        gpu_kv_argument=gpu_kv_argument, validate_args=validate_args,
        validate_workload=validate_workload, qualify_memory=qualify_memory,
        normal_install_rotation=normal_install_rotation)
    exec(compile(source, str(PKG/'run_recovery_cadence.py')+'[normal_capacity]', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
