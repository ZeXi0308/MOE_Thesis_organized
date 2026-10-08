#!/usr/bin/env python3
"""GSM8K fixed-margin recovery ablation using the qualified Qwen7B Q0 prompt.

Parent owns the shared GPU lock, verified model staging and child timeout.
Fixed sequence/batch resources and constant restore margin; no execution-backend changes.
All 1024 requests are development data in source order, with natural EOS and no answer keys.
"""
import argparse
from contextlib import nullcontext
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
import health_native as common
import native_pressure_observer_v1 as observation
import restore_fixed_margin_policy_v1 as recovery

Q0_SHA = '8afa5257e57e2a7e667e0209831626c65bd039c010b9e60a45933583ea6c54ab'
BASE = Path(__file__).resolve().parent
INPUTS = BASE / 'math_inputs1024_v1.json'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
CAP, SEED = 1024, 20260905
require, sha, dump = common.require, common.sha, common.dump
require(sha(common.__file__) == Q0_SHA, 'immutable Q0 helper source changed')
require(sha(observation.__file__) == '8ed2404a1b2546094927e9132eb47336b80e2d99e2a61ab4a2b2f678bd555877',
        'immutable native pressure observer source changed')
require(sha(recovery.__file__) == '5b881b64f321d1e3ace653178eb544209abf04e7d96f3f51f3ca3cc161e16063',
        'immutable fixed-margin policy source changed')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    engine, phase, timing = None, 'inputs', {'start_unix_s': time.time()}
    def interrupted(signum, _frame):
        raise InterruptedError('received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        require(args.kv_bytes == 8589934592 and args.max_seqs == 1024 and
                args.batch_tokens == 4096 and args.max_model_len == 4096,
                'fixed development resources differ; no KV/context pressure tuning')
        require((args.policy == 'native' and args.fixed_margin_blocks is None) or
                (args.policy == 'restore-fixed-margin' and
                 type(args.fixed_margin_blocks) is int and args.fixed_margin_blocks in (32, 48, 64)),
                'native requires no margin; restore-fixed-margin requires explicit 32/48/64 blocks')
        require(args.inputs.resolve() == INPUTS.resolve(), 'fixed math input path differs')
        source = json.loads(args.inputs.read_text())
        requests = source['requests']
        require(source['schema'] == 'c-math-scale-inputs-v1' and
                source['task'] == 'gsm8k' and len(requests) == 1024 and
                len({r['request_id'] for r in requests}) == 1024 and
                [r['example_index'] for r in requests] == list(range(1024)) and
                source['arrival_traces_s'] == [0.0] * 1024, 'fixed math inputs differ')
        sampling = source['sampling']
        for name, value in dict(temperature=0.0, max_tokens=CAP, min_tokens=0,
                               ignore_eos=False, stop=[]).items():
            require(sampling.get(name) == value, 'frozen sampling differs: ' + name)
        require(set(sampling) == {'temperature', 'max_tokens', 'min_tokens',
                                 'ignore_eos', 'stop'}, 'unexpected source sampling fields')
        for row in requests:
            require(set(row) == {'request_id', 'example_index', 'task', 'question',
                                 'chat_user_content'} and row['task'] == 'gsm8k',
                    'unexpected input fields, task, or answer-key fields')
            require(isinstance(row['question'], str) and row['question'].strip() and
                    isinstance(row['chat_user_content'], str) and
                    row['chat_user_content'].rstrip().endswith('Question: ' + row['question'].rstrip()),
                    'missing question/chat or target question differs')
        model_identity = json.loads((args.model_dir / 'MODEL_IDENTITY.json').read_text())
        require(model_identity['model_id'] == MODEL and model_identity['revision'] == REVISION,
                'math development model identity differs')
        dump(args.output / 'source-input.json', source)
        (args.output / 'cell-source.py').write_bytes(Path(__file__).read_bytes())
        dump(args.output / 'provenance.json', dict(input_sha256=sha(args.inputs),
            cell_sha256=sha(__file__), q0_helper_sha256=sha(common.__file__),
            observer_sha256=sha(observation.__file__),
            policy=args.policy, fixed_margin_blocks=args.fixed_margin_blocks,
            policy_helper_sha256=sha(recovery.__file__),
            model_dir=str(args.model_dir.resolve()), model_id=MODEL, revision=REVISION,
            answer_keys_loaded=False, generation_input_fields=['chat_user_content'],
            generation_chat_rendering='Q0 chat template plus literal Answer: cue',
            local_model_metadata_sha256={p.name: sha(p) for p in args.model_dir.iterdir()
                if p.is_file() and (p.suffix in ('.json', '.jinja', '.model') or p.name == 'merges.txt')},
            weight_integrity_scope='Parent staging verifies frozen model weight hashes.'))
        os.environ.update(CUDA_VISIBLE_DEVICES=args.expected_gpu_uuid,
            VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0',
            VLLM_USE_SIMPLE_KV_OFFLOAD='0', VLLM_USE_FLASHINFER_SAMPLER='0', HF_HUB_OFFLINE='1')
        phase = 'tokenizer'
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir),
            local_files_only=True, trust_remote_code=False)
        rendered = []
        for row in requests:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['chat_user_content']}],
                tokenize=False, add_generation_prompt=True) + 'Answer:'
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            rendered.append(dict(row, prompt=prompt, prompt_token_ids=ids, prompt_tokens=len(ids),
                prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
        dump(args.output / 'rendered-inputs.json', rendered)
        dump(args.output / 'tokenizer.json', dict(name_or_path=tokenizer.name_or_path,
            chat_template=tokenizer.chat_template, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, special_tokens_map=tokenizer.special_tokens_map,
            answer_cue='Answer:', transformers=version('transformers')))
        require(all(r['prompt_token_ids'] and r['prompt_tokens'] + CAP <= args.max_model_len
                    for r in rendered), 'prompt plus cap exceeds context; no truncation or replacement')
        phase = 'runtime'
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        torch.set_num_threads(25)
        dump(args.output / 'runtime.json', dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, vllm=vllm.__version__, transformers=version('transformers'),
            cpu_threads=torch.get_num_threads(), gpu_before=common.gpu_state(args.expected_gpu_uuid),
            runtime_environment={'VLLM_USE_FLASHINFER_SAMPLER': os.environ['VLLM_USE_FLASHINFER_SAMPLER']},
            vllm_source_sha256={n: sha(Path(vllm.__file__).parent / n) for n in common.SOURCES}))
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one visible GPU required')
        kwargs = dict(model=str(args.model_dir), tokenizer=str(args.model_dir), dtype='bfloat16',
            seed=SEED, max_model_len=args.max_model_len, max_num_seqs=args.max_seqs,
            max_num_batched_tokens=args.batch_tokens,
            kv_cache_memory_bytes=8589934592, gpu_memory_utilization=.90,
            enable_chunked_prefill=True, enable_prefix_caching=True, scheduling_policy='fcfs',
            async_scheduling=False, scheduler_reserve_full_isl=True, long_prefill_token_threshold=0,
            stream_interval=1, enforce_eager=False, enable_return_routed_experts=False,
            trust_remote_code=False, tensor_parallel_size=1, pipeline_parallel_size=1)
        dump(args.output / 'config.json', dict(engine_args=kwargs, sampling=dict(sampling, seed=SEED),
            arrival='all at zero', answer_cue='Answer:', policy=args.policy,
            fixed_margin_blocks=args.fixed_margin_blocks,
            scope='Math fixed-margin recovery ablation; all requests retained; quality analyzed separately; no held-out claim.'))
        phase = 'engine_init'; timing['engine_init_start_unix_s'] = time.time()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing['engine_init_end_unix_s'] = time.time()
        cfg = engine.vllm_config.model_config
        eos = dict(hf_eos_token_id=cfg.hf_config.to_dict().get('eos_token_id'),
            generation_config=cfg.try_get_generation_config(), generation_source=cfg.generation_config,
            generation_overrides=cfg.override_generation_config, sampling_defaults=cfg.get_diff_sampling_param())
        dump(args.output / 'resolved-eos.json', eos)
        values = eos['hf_eos_token_id']; values = [values] if type(values) is int else values
        require(isinstance(values, list) and values and all(type(v) is int and v >= 0 for v in values), 'missing EOS IDs')
        require(engine.vllm_config.cache_config.enable_prefix_caching and common.drain(engine)['status'] == 'QUALIFIED',
                'native initial/APC qualification failed')
        scheduler = engine.engine_core.engine_core.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        require(scheduler.max_num_running_reqs == args.max_seqs and
                scheduler.max_num_scheduled_tokens == args.batch_tokens and
                scheduler.max_model_len == args.max_model_len and
                scheduler.scheduler_reserve_full_isl is True and
                engine.vllm_config.cache_config.kv_cache_memory_bytes == args.kv_bytes,
                'resolved concurrency/token budget/context/KV/full-ISL differs')
        dump(args.output / 'resolved-runtime.json', dict(
            max_model_len=scheduler.max_model_len,
            max_num_running_reqs=scheduler.max_num_running_reqs,
            max_num_scheduled_tokens=scheduler.max_num_scheduled_tokens,
            kv_cache_memory_bytes=engine.vllm_config.cache_config.kv_cache_memory_bytes,
            block_size=engine.vllm_config.cache_config.block_size,
            total_blocks=pool.num_gpu_blocks, usable_blocks=pool.num_gpu_blocks - 1,
            scheduler_reserve_full_isl=scheduler.scheduler_reserve_full_isl,
            dtype=str(cfg.dtype), scope='Observed runtime geometry, not a pressure verdict.'))
        phase = 'warmup'; timing['warmup_start_unix_s'] = time.time()
        warmups = [common.generate(engine, [r], 16, label, args.output)
                   for r, label in [(rendered[0], 'warmup-first'), (rendered[-1], 'warmup-last')]]
        before = common.drain(engine); reset = bool(engine.reset_prefix_cache()); after = common.drain(engine)
        hashes = len(engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block)
        dump(args.output / 'prefix-cache-reset.json', dict(warmups=warmups, before=before,
            after=after, reset_succeeded=reset, cached_hash_keys_after=hashes))
        require(reset and not hashes and before['status'] == after['status'] == 'QUALIFIED', 'cold APC reset failed')
        timing['warmup_end_unix_s'] = time.time(); phase = 'measurement'
        timing['measurement_start_unix_s'] = time.time()
        with observation.observe_native_pressure(engine, args.output / 'measured-pressure.json'):
            policy_context = (recovery.restore_fixed_margin(engine,
                                  args.output / 'restore-fixed-margin-policy.json', args.fixed_margin_blocks)
                              if args.policy == 'restore-fixed-margin' else nullcontext())
            with policy_context:
                result = common.generate(engine, rendered, CAP, 'measured', args.output)
        timing['measurement_end_unix_s'] = time.time()
        final = common.drain(engine); dump(args.output / 'native-drain.json', final)
        require(final['status'] == 'QUALIFIED', 'native final drain failed')
        dump(args.output / 'status.json', dict(result, phase='complete', expected_requests=1024,
            policy=args.policy, fixed_margin_blocks=args.fixed_margin_blocks,
            scope='Math fixed-margin recovery ablation; all 1024 development outputs retained; no answer keys loaded.'))
    except BaseException as exc:
        dump(args.output / 'status.json', dict(status='INCOMPLETE', phase=phase,
            expected_requests=1024, policy=args.policy, fixed_margin_blocks=args.fixed_margin_blocks,
            error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        timing['shutdown_start_unix_s'] = time.time()
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        except BaseException as exc:
            dump(args.output / 'status.json', dict(status='INCOMPLETE', phase='shutdown',
                expected_requests=1024, policy=args.policy, fixed_margin_blocks=args.fixed_margin_blocks,
                error=f'{type(exc).__name__}: {exc}'))
            raise
        finally:
            timing['end_unix_s'] = time.time(); dump(args.output / 'timing.json', timing)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'inputs', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-gpu-uuid', required=True)
    p.add_argument('--policy', choices=('native', 'restore-fixed-margin'), required=True)
    p.add_argument('--fixed-margin-blocks', type=int, choices=(32, 48, 64),
                   help='Required for restore-fixed-margin; omit for native.')
    p.add_argument('--kv-bytes', type=int, default=8589934592)
    p.add_argument('--max-seqs', type=int, choices=(1024,), default=1024)
    p.add_argument('--batch-tokens', type=int, choices=(4096,), default=4096)
    p.add_argument('--max-model-len', type=int, default=4096)
    args = p.parse_args()
    if (args.policy == 'native') != (args.fixed_margin_blocks is None):
        p.error('native requires no --fixed-margin-blocks; restore-fixed-margin requires it')
    run(args)
