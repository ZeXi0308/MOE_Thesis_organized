#!/usr/bin/env python3
"""Independent native, synchronous 16-request model/workload qualification.

The parent owns the shared GPU lock and process timeout. No policy comparison.
Only standard-library imports occur before argument parsing / --help.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CAP, SEED = 1024, 20260905
SOURCES = ['v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py',
           'v1/core/block_pool.py', 'v1/worker/gpu_model_runner.py',
           'v1/core/kv_cache_coordinator.py', 'v1/core/single_type_kv_cache_manager.py',
           'v1/core/kv_cache_utils.py', 'config/model.py']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + '\n')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def gpu_state(expected):
    def query(kind, fields):
        return subprocess.check_output(['nvidia-smi', '-i', expected,
            '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'],
            text=True, timeout=15).strip()
    info = query('gpu', 'uuid,name,memory.total,memory.used')
    require(info.split(',')[0].strip() == expected, 'GPU identity differs')
    processes = query('compute-apps', 'pid,used_gpu_memory')
    require(all(int(line.split(',')[0]) == os.getpid()
                for line in processes.splitlines() if line.strip()),
            'another process uses the selected GPU')
    return dict(gpu=info, compute_processes=processes)


def drain(engine):
    s = engine.engine_core.engine_core.scheduler
    pool = s.kv_cache_manager.block_pool
    running, waiting = s.get_request_counts()
    row = dict(unfinished=bool(engine.has_unfinished_requests()),
        requests=len(s.requests), running=running, waiting=waiting,
        total_blocks=pool.num_gpu_blocks, free_blocks=pool.get_num_free_blocks(),
        connector_absent=s.connector is None,
        kv_transfer_absent=engine.vllm_config.kv_transfer_config is None,
        offload=getattr(engine.vllm_config.cache_config, 'kv_offloading_size', None))
    row['status'] = 'QUALIFIED' if (not row['unfinished'] and
        row['requests'] == row['running'] == row['waiting'] == 0 and
        row['total_blocks'] > 1 and row['free_blocks'] == row['total_blocks'] - 1 and
        row['connector_absent'] and row['kv_transfer_absent'] and row['offload'] is None) else 'UNQUALIFIED'
    return row


def generate(engine, requests, cap, label, output):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    s = engine.engine_core.engine_core.scheduler
    original = s.schedule
    preempted, steps, native_sampling = [], [], {}
    rows = {label + '/' + r['request_id']: dict(r, external_request_id=label + '/' + r['request_id'],
        output_text='', output_token_ids=[], token_times_s=[], host_returns=[],
        finished=False, finish_reason=None, stop_reason=None, arrival_s=0.0) for r in requests}
    start, epoch = time.perf_counter(), time.time()
    def schedule(*args, **kwargs):
        if not native_sampling:
            for rid, request in s.requests.items():
                params = request.sampling_params
                native_sampling[rid] = {name: getattr(params, name, None) for name in
                    ('stop', 'stop_token_ids', 'ignore_eos', 'min_tokens', 'max_tokens', '_eos_token_id')}
                native_sampling[rid]['all_stop_token_ids'] = sorted(getattr(params, '_all_stop_token_ids', ()))
        result = original(*args, **kwargs)
        preempted.extend(result.preempted_req_ids or [])
        return result
    s.schedule = schedule
    try:
        with (output / (label + '-host-returns.jsonl')).open('x') as journal:
            for rid, row in rows.items():
                row['add_request_s'] = time.perf_counter() - start
                engine.add_request(rid, {'prompt_token_ids': row['prompt_token_ids']},
                    SamplingParams(n=1, temperature=0.0, max_tokens=cap, min_tokens=0,
                        ignore_eos=False, stop=[], stop_token_ids=[], detokenize=True,
                        output_kind=RequestOutputKind.CUMULATIVE), arrival_time=epoch)
            while engine.has_unfinished_requests():
                before = time.perf_counter() - start
                require(before <= 900, 'generation exceeds 900 seconds')
                outputs = engine.step()
                received = time.perf_counter() - start
                pool = s.kv_cache_manager.block_pool
                steps.append(dict(start_s=before, return_s=received, output_requests=len(outputs),
                    used_blocks_after_step=pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()))
                for item in outputs:
                    row = rows[item.request_id]
                    require(not row['finished'] and len(item.outputs) == 1, 'unexpected output/completion')
                    completion = item.outputs[0]
                    tokens, old = list(completion.token_ids), row['output_token_ids']
                    require(tokens[:len(old)] == old and len(old) <= len(tokens) <= cap,
                            'cumulative token prefix/cap violated')
                    event = dict(request_id=item.request_id, return_s=received,
                        delta_token_ids=tokens[len(old):], cumulative_tokens=len(tokens),
                        finished=bool(item.finished), finish_reason=completion.finish_reason,
                        stop_reason=getattr(completion, 'stop_reason', None))
                    journal.write(json.dumps(event, ensure_ascii=False) + '\n')
                    row['host_returns'].append(event)
                    row['token_times_s'].extend([received] * (len(tokens) - len(old)))
                    row.update(output_token_ids=tokens, output_text=completion.text,
                        finished=event['finished'], finish_reason=event['finish_reason'],
                        stop_reason=event['stop_reason'])
                    if item.finished:
                        require(completion.finish_reason in ('stop', 'length'), 'unexpected finish reason')
                        row['host_elapsed_s'] = received
                journal.flush()
            require(all(r['finished'] for r in rows.values()), 'unfinished planned requests')
        return dict(status='COMPLETE', request_count=len(rows), preemptions=len(preempted),
            observation_end_s=time.perf_counter() - start,
            output_tokens=sum(len(r['output_token_ids']) for r in rows.values()),
            finish_reason_counts=dict(Counter(r['finish_reason'] for r in rows.values())))
    finally:
        s.schedule = original
        dump(output / (label + '-native-sampling.json'), native_sampling)
        dump(output / (label + '-outputs.json'), list(rows.values()))
        dump(output / (label + '-steps.json'), dict(steps=steps, preempted_request_ids=preempted,
            timing='Host return timestamps; simultaneous delivery is not per-token device ITL.'))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    engine, timing, phase = None, {'start_unix_s': time.time()}, 'inputs'
    def interrupted(signum, _frame):
        raise InterruptedError('received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        require(args.kv_bytes > 0 and args.max_seqs > 0 and args.batch_tokens > 0 and
                args.max_model_len > CAP, 'invalid capacity/configuration')
        input_file = args.inputs / 'workload.json' if args.inputs.is_dir() else args.inputs
        source = json.loads(input_file.read_text())
        requests = source['requests']
        require(len(requests) == 16 and len({r['request_id'] for r in requests}) == 16,
                'expected the fixed sixteen distinct requests')
        require([r['example_index'] for r in requests] == list(range(16)), 'source order differs')
        require(source.get('arrival_traces_s') == [0.0] * 16, 'qualification arrivals differ')
        dump(args.output / 'source-input.json', source)
        (args.output / 'cell-source.py').write_bytes(Path(__file__).read_bytes())
        provenance = dict(input_path=str(input_file.resolve()), input_sha256=sha(input_file),
            cell_sha256=sha(__file__), model_dir=str(args.model_dir.resolve()))
        receipt = input_file.parent / 'SOURCE_RECEIPT.json'
        if receipt.exists():
            provenance['input_source_receipt'] = json.loads(receipt.read_text())
        provenance['local_model_files'] = {p.name: dict(bytes=p.stat().st_size,
            sha256=sha(p) if p.suffix != '.safetensors' else None)
            for p in args.model_dir.iterdir() if p.is_file() and
            (p.suffix in ('.json', '.jinja', '.model', '.safetensors') or p.name == 'merges.txt')}
        provenance['weight_integrity_scope'] = 'Weight sizes only here; parent staging must verify frozen weight hashes.'
        dump(args.output / 'provenance.json', provenance)
        os.environ.update(CUDA_VISIBLE_DEVICES=args.expected_gpu_uuid,
            VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0',
            VLLM_USE_SIMPLE_KV_OFFLOAD='0', HF_HUB_OFFLINE='1')
        phase = 'tokenizer'
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True,
                                                  trust_remote_code=False)
        rendered = []
        for row in requests:
            require(isinstance(row.get('chat_user_content'), str) and row['chat_user_content'] and
                    isinstance(row.get('gold'), str), 'missing chat content or gold')
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['chat_user_content']}],
                tokenize=False, add_generation_prompt=True) + 'Answer:'
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            rendered.append(dict(row, prompt=prompt, prompt_token_ids=ids, prompt_tokens=len(ids),
                source_prompt_token_ids_sha256=row.get('prompt_token_ids_sha256'),
                prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
        dump(args.output / 'rendered-inputs.json', rendered)
        dump(args.output / 'tokenizer.json', dict(name_or_path=tokenizer.name_or_path,
            chat_template=tokenizer.chat_template, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, special_tokens_map=tokenizer.special_tokens_map,
            answer_cue='Answer:', transformers=version('transformers')))
        require(all(r['prompt_token_ids'] and r['prompt_tokens'] + CAP <= args.max_model_len
                    for r in rendered), 'prompt plus output cap exceeds context; no truncation')
        phase = 'runtime'
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        torch.set_num_threads(25)
        runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version('transformers'), cpu_threads=torch.get_num_threads(),
            vllm_source_sha256={n: sha(Path(vllm.__file__).parent / n) for n in SOURCES},
            gpu_before=gpu_state(args.expected_gpu_uuid))
        dump(args.output / 'runtime.json', runtime)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one visible CUDA GPU required')
        kwargs = dict(model=str(args.model_dir), tokenizer=str(args.model_dir), dtype='bfloat16',
            seed=SEED, max_model_len=args.max_model_len, max_num_seqs=args.max_seqs,
            max_num_batched_tokens=args.batch_tokens, kv_cache_memory_bytes=args.kv_bytes,
            gpu_memory_utilization=0.90, enable_chunked_prefill=True, enable_prefix_caching=True,
            scheduling_policy='fcfs', async_scheduling=False, scheduler_reserve_full_isl=True,
            long_prefill_token_threshold=0, stream_interval=1, enforce_eager=False,
            enable_return_routed_experts=False, trust_remote_code=False)
        kwargs.update(tensor_parallel_size=1, pipeline_parallel_size=1)
        dump(args.output / 'config.json', dict(engine_args=kwargs, sampling=dict(temperature=0.0,
            max_tokens=CAP, min_tokens=0, ignore_eos=False, stop=[], seed=SEED), arrival='all at zero',
            scope='Native workload/model health qualification; no policy comparison.'))
        phase = 'engine_init'; timing['engine_init_start_unix_s'] = time.time()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing['engine_init_end_unix_s'] = time.time()
        cfg = engine.vllm_config.model_config
        eos = dict(hf_eos_token_id=cfg.hf_config.to_dict().get('eos_token_id'),
            generation_config=cfg.try_get_generation_config(), generation_source=cfg.generation_config,
            generation_overrides=cfg.override_generation_config, sampling_defaults=cfg.get_diff_sampling_param())
        dump(args.output / 'resolved-eos.json', eos)
        values = eos['hf_eos_token_id']; values = [values] if type(values) is int else values
        require(isinstance(values, list) and values and all(type(v) is int and v >= 0 for v in values),
                'missing model EOS IDs')
        require(engine.vllm_config.cache_config.enable_prefix_caching and drain(engine)['status'] == 'QUALIFIED',
                'native initial/APC qualification failed')
        phase = 'warmup'; timing['warmup_start_unix_s'] = time.time()
        warmups = [generate(engine, [r], 16, label, args.output)
                   for r, label in [(rendered[0], 'warmup-first'), (rendered[-1], 'warmup-last')]]
        before = drain(engine); reset = bool(engine.reset_prefix_cache()); after = drain(engine)
        hashes = len(engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block)
        dump(args.output / 'prefix-cache-reset.json', dict(warmups=warmups, before=before, after=after,
             reset_succeeded=reset, cached_hash_keys_after=hashes))
        require(reset and not hashes and before['status'] == after['status'] == 'QUALIFIED', 'cold APC reset failed')
        timing['warmup_end_unix_s'] = time.time(); phase = 'measurement'
        timing['measurement_start_unix_s'] = time.time()
        result = generate(engine, rendered, CAP, 'measured', args.output)
        timing['measurement_end_unix_s'] = time.time()
        final = drain(engine); dump(args.output / 'native-drain.json', final)
        require(final['status'] == 'QUALIFIED', 'native final drain failed')
        dump(args.output / 'status.json', dict(result, phase='complete', expected_requests=16,
            scope='Health qualification only; quality analysis uses all sixteen planned requests.'))
    except BaseException as exc:
        dump(args.output / 'status.json', dict(status='INCOMPLETE', phase=phase,
            expected_requests=16, error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        timing['shutdown_start_unix_s'] = time.time()
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        except BaseException as exc:
            dump(args.output / 'status.json', dict(status='INCOMPLETE', phase='shutdown',
                expected_requests=16, error=f'{type(exc).__name__}: {exc}'))
            raise
        finally:
            timing['end_unix_s'] = time.time(); dump(args.output / 'timing.json', timing)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'inputs', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-gpu-uuid', required=True)
    p.add_argument('--kv-bytes', type=int, required=True)
    p.add_argument('--max-seqs', type=int, default=32)
    p.add_argument('--batch-tokens', type=int, default=1024)
    p.add_argument('--max-model-len', type=int, default=4096)
    run(p.parse_args())
