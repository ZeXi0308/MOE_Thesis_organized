"""Real native continuous-service experiment with bounded same-engine arms."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback
from selector import RestoreSelector, validate_target_spec
from native_transfer_probe import NativeTransferStartProbe
from host_history_observer import HostHistoryObserver
from native_store_replay import NativeStoreReplay
from native_inline_host import NativeInlineHostRecovery


def telemetry_arm_modes(spec, count, timing_enabled):
    """Only toggle the passive GPU child; keep step tracing/CPU clocks unchanged."""
    if spec is None:
        return ['on' if timing_enabled else 'off'] * count
    modes = spec.split(',')
    if not timing_enabled or len(modes) != count or any(x not in ('on', 'off') for x in modes):
        raise ValueError('telemetry modes require --timing-observer and one on/off value per arm')
    return modes


def store_replay_arm_modes(spec, count):
    """Explicit one-entry writeback intervention; never changes warmup."""
    if spec is None:
        return ['off'] * count
    modes = spec.split(',')
    if len(modes) != count or any(x not in ('on', 'off') for x in modes):
        raise ValueError('store replay modes require one on/off value per arm')
    return modes


def host_load_arm_modes(spec, count):
    modes = ['async'] * count if spec is None else spec.split(',')
    if len(modes) != count or any(x not in ('async', 'inline') for x in modes):
        raise ValueError('host load modes require one async/inline value per arm')
    return modes


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def host_memory():
    paths = ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current',
             '/sys/fs/cgroup/memory.events','/proc/meminfo']
    return {name:Path(name).read_text() for name in paths if Path(name).is_file()}


def drain(engine):
    start = time.perf_counter()
    while engine.engine_core.engine_core.scheduler.connector.has_pending_push_work():
        assert time.perf_counter() - start < 180
        assert not engine.step()


def check_isolation(path):
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,temperature.gpu,power.draw,clocks.current.sm','--format=csv,noheader'],text=True)
    procs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    pids = [int(p.strip()) for p in procs.splitlines() if p.strip()]
    dump(path,dict(time=time.time(),gpu=gpu,compute_pids=pids,allowed_pid=os.getpid()))
    assert os.environ['E_EXPECTED_GPU_UUID'] in gpu and set(pids) <= {os.getpid()}


def reset_drained(engine, selector, expected_free_blocks):
    drain(engine)
    assert engine.reset_prefix_cache(reset_connector=True)
    drain(engine)
    s, cs = selector.scheduler, selector.cs
    assert not s.requests and not s.running and not s.waiting
    assert not s.skipped_waiting and not s._inflight_prefills
    assert not cs._jobs and not cs._req_status
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == expected_free_blocks
    return dict(reset_connector=True, success=True, free_blocks=expected_free_blocks,
                pending_jobs=0, pending_requests=0)


def episode(engine, selector, inputs, output_tokens, interval, natural, name, max_seconds,
            timing_observer=False, transfer_probe=None, host_history=None, store_replay=None,
            inline_host=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    scheduler = engine.engine_core.engine_core.scheduler
    rows = []
    by_external = {}
    last_scheduled = {}
    steps = []
    schedule_trace = []
    native_schedule = scheduler.schedule
    def observed_schedule(*args, **kwargs):
        result = native_schedule(*args, **kwargs)
        scheduled = []
        last_scheduled.clear()
        for rid, count in result.num_scheduled_tokens.items():
            req = scheduler.requests[rid]
            scheduled.append(dict(request_id=rid, count=count,
                start_computed=req.num_computed_tokens-count,
                end_computed=req.num_computed_tokens, known_tokens=req.num_tokens,
                generated_tokens=req.num_output_tokens))
            last_scheduled[rid] = scheduled[-1]
        schedule_trace.append(dict(time_s=selector.now(), scheduled=scheduled,
            free_blocks_after_schedule=scheduler.kv_cache_manager.block_pool.get_num_free_blocks()))
        return result
    scheduler.schedule = observed_schedule
    selector.clear()
    clock_alignment = dict(episode_origin_monotonic_s=selector.origin,
                           alignment_monotonic_s=time.perf_counter(),
                           alignment_unix_s=time.time())
    selector.partial = dict(status='INCOMPLETE', episode=name, requests=rows, steps=steps,
        scheduler_steps=schedule_trace, decisions=selector.events, commits=selector.commits,
        preemptions=selector.preemptions, transfers=selector.transfers,
        clock_alignment=clock_alignment)
    selector.partial['host_history_observer'] = dict(
        enabled=host_history is not None and host_history.enabled,
        records=host_history.records if host_history is not None else [],
        errors=host_history.errors if host_history is not None else [])
    selector.partial['store_replay'] = dict(
        enabled=store_replay is not None and store_replay.enabled,
        records=store_replay.records if store_replay is not None else [],
        errors=store_replay.errors if store_replay is not None else [],
        executed_count=0)
    selector.partial['inline_host'] = dict(
        enabled=inline_host is not None and inline_host.enabled,
        records=inline_host.records if inline_host is not None else [])
    if transfer_probe is not None:
        selector.partial['native_transfer_start_calls'] = transfer_probe.records
    epoch = time.time()
    arrivals = [float(source.get('arrival_s', index * interval)) for index, source in enumerate(inputs)]
    assert arrivals == sorted(arrivals) and all(x >= 0 for x in arrivals)
    index = 0
    while index < len(inputs) or engine.has_unfinished_requests():
        now = selector.now()
        if now > max_seconds:
            raise TimeoutError(f'{name} exceeded {max_seconds}s')
        while index < len(inputs) and arrivals[index] <= now:
            source = inputs[index]
            cap = int(source.get('output_tokens', output_tokens))
            external = f'{name}/E{index:03d}'
            params = SamplingParams(temperature=0.0, max_tokens=cap,
                min_tokens=0 if natural else cap, ignore_eos=not natural,
                detokenize=False, output_kind=RequestOutputKind.CUMULATIVE)
            admitted = selector.now()
            internal = engine.add_request(external, dict(prompt_token_ids=source['prompt_token_ids'],
                cache_salt=f'E-{name}-{index}'), params, arrival_time=epoch+arrivals[index])
            if selector.target_spec is not None:
                selector.register_request(internal, external)
            rows.append(dict(request_id=internal, external_id=external, arrival_s=arrivals[index],
                admitted_s=admitted, prompt_tokens=len(source['prompt_token_ids']),
                max_output_tokens=cap,
                source_index=source.get('example_index'), gold=source.get('gold'),
                output_token_ids=[], token_times_s=[], completed=False))
            by_external[external] = rows[-1]
            index += 1
        if not engine.has_unfinished_requests():
            time.sleep(min(0.001, max(0, arrivals[index]-selector.now())))
            continue
        if transfer_probe is not None:
            transfer_probe.service_step(len(steps))
        if host_history is not None:
            host_history.service_step(len(steps))
        if store_replay is not None:
            store_replay.service_step(len(steps))
        if inline_host is not None:
            inline_host.service_step(len(steps))
        before = selector.now()
        if timing_observer:
            process_before, thread_before = time.process_time(), time.thread_time()
        outputs = engine.step()
        end = selector.now()
        if timing_observer:
            process_after, thread_after = time.process_time(), time.thread_time()
        selector.last_step_s = end-before
        steps.append(dict(start_s=before, end_s=end, running=len(scheduler.running),
            waiting=len(scheduler.waiting), free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            pending_load_jobs=sum(not j.is_store for j in selector.cs._jobs.values()),
            host_resident_blocks=selector.cs.manager._num_allocated_blocks-len(selector.cs.manager._free_list)))
        if timing_observer:
            steps[-1].update(process_cpu_s=process_after-process_before,
                             driver_thread_cpu_s=thread_after-thread_before)
        for output in outputs:
            row = by_external[output.request_id]
            assert len(output.outputs) == 1 and not row['completed']
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            previous = row['output_token_ids']
            assert tokens[:len(previous)] == previous and len(previous) <= len(tokens) <= row['max_output_tokens']
            if len(tokens) > len(previous):
                scheduled = last_scheduled[row['request_id']]
                assert scheduled['end_computed'] >= scheduled['known_tokens'], 'recompute chunk emitted new output early'
            row['token_times_s'].extend([end]*(len(tokens)-len(previous)))
            row['output_token_ids'] = tokens
            if output.finished:
                row.update(completed=True, completion_s=end, finish_reason=completion.finish_reason,
                           stop_reason=completion.stop_reason)
                if not natural:
                    assert len(tokens) == row['max_output_tokens'] and completion.finish_reason == 'length'
    complete_s = selector.now()
    assert len(rows) == len(inputs) and all(r['completed'] for r in rows)
    if transfer_probe is not None:
        transfer_probe.drain()
    if host_history is not None:
        host_history.drain()
    if store_replay is not None:
        store_replay.drain()
    if inline_host is not None:
        inline_host.drain()
    drain(engine)
    scheduler.schedule = native_schedule
    if store_replay is not None:
        selector.partial['store_replay']['executed_count'] = store_replay.executed_count
    return dict(requests=rows, steps=steps, decisions=selector.events, commits=selector.commits,
        scheduler_steps=schedule_trace,
        preemptions=selector.preemptions, transfers=selector.transfers, all_complete_s=complete_s,
        service_and_drain_s=selector.now(), prefix_consistency=True,
        clock_alignment=clock_alignment, target_selection=selector.target_summary(),
        host_history_observer=selector.partial['host_history_observer'],
        store_replay=selector.partial['store_replay'],
        inline_host=selector.partial['inline_host'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--policy', choices=['host','recompute','length','headroom'], required=True)
    p.add_argument('--threshold', type=int, default=0)
    p.add_argument('--model', required=True)
    p.add_argument('--workload', type=Path, required=True)
    p.add_argument('--requests', type=int, default=64)
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--interval', type=float, default=0)
    p.add_argument('--kv-bytes', type=int, default=None, help='Omit for normal native profiling; explicit limits are diagnostics')
    p.add_argument('--host-gib', type=int, default=32)
    p.add_argument('--max-num-seqs', type=int, default=256)
    p.add_argument('--batch-tokens', type=int, default=2048)
    p.add_argument('--policies', help='Bounded same-engine arms, e.g. host,recompute,recompute,host')
    p.add_argument('--natural', action='store_true')
    p.add_argument('--max-seconds', type=float, default=600)
    p.add_argument('--timing-observer', action='store_true',
                   help='Passive 1 Hz GPU snapshots and per-step CPU time; no policy input')
    p.add_argument('--native-transfer-probe', action='store_true',
                   help='Passive native transfer-start wall boundary; includes deferred STORE, not pure LOAD time')
    p.add_argument('--host-history-observer', action='store_true',
                   help='Passive native STORE cursor/job fields; identical in warmup and every arm')
    p.add_argument('--store-replay-modes',
                   help='Causal one-event R-only probe: off/on per formal arm; warmup always off')
    p.add_argument('--host-load-modes',
                   help='Native Host async/inline per arm; complete storable prefix only, warmup always async')
    p.add_argument('--gpu-telemetry-modes',
                   help='Diagnostic only: comma-separated on/off per arm; per-step CPU timing stays enabled')
    p.add_argument('--telemetry-script', type=Path,
                   default=Path(__file__).with_name('timing_observer.py'),
                   help='Explicit passive observer source; recorded and hashed even in off arms')
    p.add_argument('--target-spec', type=Path,
                   help='Optional frozen one-event target; all other recoveries retain native Host policy')
    a = p.parse_args()
    policies = a.policies.split(',') if a.policies else [a.policy]
    assert all(policy in ('host','recompute','length','headroom') for policy in policies)
    telemetry_modes = telemetry_arm_modes(a.gpu_telemetry_modes, len(policies), a.timing_observer)
    replay_modes = store_replay_arm_modes(a.store_replay_modes, len(policies))
    host_modes = host_load_arm_modes(a.host_load_modes, len(policies))
    if 'inline' in host_modes:
        assert all(policy == 'host' for policy in policies)
        assert a.target_spec is None and 'on' not in replay_modes
    a.telemetry_script = a.telemetry_script.resolve()
    if a.timing_observer:
        assert a.telemetry_script.is_file()
    target_spec = json.loads(a.target_spec.read_text()) if a.target_spec else None
    validate_target_spec(target_spec)
    if 'on' in replay_modes:
        assert target_spec is not None and a.host_history_observer
        assert all(policy == 'recompute' for policy in policies)
    if target_spec is not None:
        assert all(policy in ('host','recompute') for policy in policies)
        assert hashlib.sha256(a.workload.read_bytes()).hexdigest() == target_spec['source']['workload_sha256']
    a.out.mkdir(parents=True, exist_ok=False)
    dump(a.out/'config.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    if target_spec is not None:
        dump(a.out/'target_spec.json', target_spec)
    engine = None
    try:
        dump(a.out/'host_memory_before.json',host_memory())
        import torch
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        source = json.loads(a.workload.read_text())['source_requests']
        inputs = source[:a.requests]
        assert len(inputs) == a.requests
        dump(a.out/'inputs.json', inputs)
        kwargs = dict(model=a.model, tokenizer=a.model, dtype='bfloat16', seed=20261004,
            max_model_len=4096, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.batch_tokens,
            kv_cache_memory_bytes=a.kv_bytes, gpu_memory_utilization=0.9,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            scheduling_policy='fcfs', async_scheduling=False, stream_interval=1,
            enforce_eager=True, kernel_config={'moe_backend':'triton'},
            kv_offloading_size=a.host_gib, kv_offloading_backend='native',
            kv_transfer_config={'kv_connector_extra_config':{'offload_prompt_only':False}})
        dump(a.out/'engine_args.json', kwargs)
        dump(a.out/'versions.json', {x:importlib.metadata.version(x) for x in ['vllm','torch','transformers']})
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        dump(a.out/'host_memory_after_init.json',host_memory())
        scheduler = engine.engine_core.engine_core.scheduler
        scheduler.scheduler_reserve_full_isl = True
        selector = RestoreSelector(scheduler, a.policy, a.threshold, target_spec=target_spec)
        worker_connector = sys.modules['vllm.distributed.kv_transfer.kv_transfer_state']._KV_CONNECTOR_AGENT
        source_paths = [Path(inspect.getfile(type(obj))) for obj in
            (scheduler, selector.cs, selector.cs.manager, scheduler.kv_cache_manager)]
        source_paths += [Path(inspect.getfile(type(obj))) for obj in
            (worker_connector, worker_connector.connector_worker,
             worker_connector.connector_worker.worker)]
        source_paths += [Path(a.model)/'config.json', a.workload,
                         Path(__file__), Path(__file__).with_name('selector.py'),
                         Path(__file__).with_name('native_transfer_probe.py'),
                         Path(__file__).with_name('host_history_observer.py'),
                         Path(__file__).with_name('native_store_replay.py')]
        source_paths.append(Path(__file__).with_name('native_inline_host.py'))
        if a.timing_observer:
            source_paths.append(a.telemetry_script)
        if a.target_spec:
            assert json.loads(a.target_spec.read_text()) == target_spec, 'Target spec changed during initialization'
            source_paths.append(a.target_spec)
        dump(a.out/'runtime_source_hashes.json', {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in source_paths})
        pool = scheduler.kv_cache_manager.block_pool
        runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        worker = worker_connector.connector_worker.worker
        host_tensors = list(worker._store_handler.dst_tensors) + list(worker._load_handler.src_tensors)
        def storage_bytes(tensors):
            return sum({(str(t.device), t.untyped_storage().data_ptr()):t.untyped_storage().nbytes()
                        for t in tensors}.values())
        actual_gpu_bytes = storage_bytes(runner.kv_caches)
        actual_host_bytes = storage_bytes(host_tensors)
        if a.kv_bytes is not None:
            assert actual_gpu_bytes == a.kv_bytes
        assert actual_host_bytes == a.host_gib*1024**3
        assert all(t.is_pinned() for t in host_tensors)
        resources = dict(gpu_blocks=pool.num_gpu_blocks,
            free_gpu_blocks=pool.get_num_free_blocks(), host_blocks=selector.cs.manager._num_blocks,
            group_config=[dict(tokens_per_block=c.tokens_per_block,tokens_per_chunk=c.tokens_per_chunk)
                for c in selector.cs.config.kv_group_configs], full_sequence_must_fit=True,
            gpu_bytes=actual_gpu_bytes, host_bytes=actual_host_bytes, host_all_pinned=True,
            capacity_mode='native_profile_0.9' if a.kv_bytes is None else 'controlled_limited_capacity')
        dump(a.out/'resources.json', resources)
        completed = []
        target_results = {}
        for index, policy in enumerate(policies):
            replay = None  # A later warmup failure must not inherit the prior arm's probe.
            active_out = a.out/f'{index:02d}_{policy}' if a.policies else a.out
            gpu_telemetry_enabled = telemetry_modes[index] == 'on'
            store_replay_enabled = replay_modes[index] == 'on'
            inline_host_enabled = host_modes[index] == 'inline'
            if a.policies:
                active_out.mkdir(exist_ok=False)
                config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
                config.update(policy=policy, out=str(active_out), gpu_telemetry_enabled=gpu_telemetry_enabled,
                              store_replay_enabled=store_replay_enabled,
                              host_load_mode=host_modes[index])
                dump(active_out/'config.json',config)
                dump(active_out/'engine_args.json',kwargs)
                dump(active_out/'inputs.json',inputs)
                dump(active_out/'resources.json',resources)
                if target_spec is not None:
                    dump(active_out/'target_spec.json',target_spec)
            else:
                config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
                config['gpu_telemetry_enabled'] = gpu_telemetry_enabled
                config['store_replay_enabled'] = store_replay_enabled
                config['host_load_mode'] = host_modes[index]
                dump(active_out/'config.json', config)
            selector.policy = policy
            dump(a.out/'status.json',dict(status='RUNNING',cell=active_out.name,completed=completed))
            # Every arm pays the same full-pressure native-Host warmup. Reset
            # BEFORE warmup as well as after it prevents prior-arm cache reuse.
            selector.enabled = False
            check_isolation(active_out/'gpu_before_warmup.json')
            reset_drained(engine, selector, resources['free_gpu_blocks'])
            with HostHistoryObserver(selector.cs, selector.now, selector,
                                     enabled=a.host_history_observer) as history:
                warm = episode(engine, selector, inputs, a.output_tokens, a.interval,
                               False, 'warm', a.max_seconds,
                               host_history=history if a.host_history_observer else None)
            dump(active_out/'warmup.json', warm)
            reset = reset_drained(engine, selector, resources['free_gpu_blocks'])
            dump(active_out/'warmup_reset.json',reset)
            check_isolation(active_out/'gpu_before_measurement.json')
            selector.enabled = True
            observer = None
            with (active_out/'timing_observer.jsonl').open('x') as observer_log:
                try:
                    if gpu_telemetry_enabled:
                        observer = subprocess.Popen([sys.executable,
                            str(a.telemetry_script),
                            '--parent-pid', str(os.getpid())], stdout=observer_log,
                            stderr=subprocess.STDOUT)
                    with NativeTransferStartProbe(worker_connector, selector.now,
                                                  a.native_transfer_probe) as probe, \
                         NativeStoreReplay(selector, enabled=store_replay_enabled) as replay, \
                         HostHistoryObserver(selector.cs, selector.now, selector,
                                             enabled=a.host_history_observer) as history, \
                         NativeInlineHostRecovery(selector, worker_connector,
                                                  enabled=inline_host_enabled) as inline_host:
                        data = episode(engine, selector, inputs, a.output_tokens, a.interval,
                                       a.natural, 'measured', a.max_seconds, a.timing_observer,
                                       probe if a.native_transfer_probe else None,
                                       history if a.host_history_observer else None,
                                       replay if store_replay_enabled else None,
                                       inline_host if inline_host_enabled else None)
                        if a.native_transfer_probe:
                            data['native_transfer_start_calls'] = probe.records
                finally:
                    if observer is not None:
                        observer.terminate()
                        observer.wait(timeout=10)
            dump(active_out/'raw.json', data)
            check_isolation(active_out/'gpu_after_measurement.json')
            dump(active_out/'status.json', dict(status='COMPLETE', requests=len(data['requests']),
                committed_recoveries=len(data['commits']), eligible_commits=sum(x['eligible'] for x in data['commits']),
                **selector.target_summary()))
            target_results[active_out.name] = selector.target_summary()
            completed.append(active_out.name)
            dump(active_out/'host_memory_after.json',host_memory())
        if a.policies:
            dump(a.out/'status.json',dict(status='COMPLETE',completed=completed,target_results=target_results))
    except BaseException as exc:
        if 'selector' in locals() and hasattr(selector, 'partial'):
            if 'replay' in locals() and replay is not None and 'store_replay' in selector.partial:
                selector.partial['store_replay']['executed_count'] = replay.executed_count
            selector.partial['target_selection'] = selector.target_summary()
            dump((active_out if 'active_out' in locals() else a.out)/'partial_raw.json', selector.partial)
        failure = dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        if 'selector' in locals():
            failure.update(selector.target_summary())
        if 'active_out' in locals() and active_out != a.out:
            dump(active_out/'status.json',failure)
        dump(a.out/'status.json',failure)
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
