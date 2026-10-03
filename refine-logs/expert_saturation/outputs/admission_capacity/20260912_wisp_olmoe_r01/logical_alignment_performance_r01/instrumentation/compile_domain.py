"""Compile/load the fixed BF16 MoE domain without launching kernels or touching pager state."""
import hashlib
import importlib
import json
from pathlib import Path
import time
import traceback

EXPECTED = {
    'fused_moe': 'a8015d90908883d3dc459e7a508d2de56bfbdff678c5f36e23d0410ef5a04683',
    'moe_align_block_size': 'c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b',
}


def assignment(m, block, global_experts, mapped):
    naive = not mapped and m * 8 * 4 <= global_experts
    length = m * 8 + global_experts * (block - 1)
    if m * 8 < global_experts:
        length = min(length, m * 8 * block)
    return naive, length


def plain(value):
    if value is None or type(value) in (str, int, float, bool): return value
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if type(value).__module__.startswith('triton'): return str(value)
    raise TypeError('unexpected non-metadata value: ' + type(value).__name__)


def snapshot(runtime):
    layers = {}
    for entry in sorted(runtime.layers.values(), key=lambda e: e['layer_name']):
        state = entry['state']; last = state.last_topk_ids_cpu
        if last is not None and last.device.type != 'cpu':
            raise RuntimeError('last_topk_ids_cpu is not on CPU')
        layers[entry['layer_name']] = dict(cap=state.cap_experts,
            slots=list(state.slot_to_expert), mapping=sorted(state.expert_to_slot.items()),
            lru=list(state.lru_tick), clock=state.lru_clock,
            last_topk=None if last is None else last.tolist(), last_unique=list(state.last_unique_experts),
            ever_loaded=sorted(entry['ever_loaded']),
            stats={name: getattr(state, name) for name in (
                'stats_forward', 'stats_miss', 'stats_evict',
                'stats_hits', 'stats_pred_hits', 'stats_pred_total')},
            tensors={n: dict(pointer=t.data_ptr(), shape=list(t.shape))
                     for n in ('scratch_w13', 'scratch_w2', 'expert_map_device', 'cpu_w13', 'cpu_w2')
                     if (t := getattr(state, n)) is not None})
    return dict(layers=layers, records=len(runtime.records), events=len(runtime.events),
                flushed_calls=runtime.flushed_calls, context=dict(runtime.context), measurement=runtime.measurement)


def memory(torch, device):
    return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device))


def precompile(runtime, mode, output_path, logical_alignment=False):
    """After engine initialization, before request warmup; output_path is an exclusive JSON file."""
    start = time.perf_counter_ns(); destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='moe_compile_domain_v1', status='STARTED', start_perf_ns=start,
        requested_mode=mode, logical_alignment=logical_alignment, expected_invocations=320, calls=[], sources={},
        scope='Fixed M=1..160, two BF16 GEMMs, actual mode weight layout. Compile and initialize handles only; '
              'no kernel launches, expert copies, pager ensure, KV writes, synchronization or peak reset. '
              'Covers this MoE Triton family only; other runtime warmup remains required. '
              'Setup time includes allocation and release; final JSON write is outside its own interval. '
              'Full process cost must retain setup and output. CUDA peaks are since prior owner reset.')
    buffers = {}; original = kernel = device = None; had_run = patched = False; own_run = None
    with destination.open('x') as stream:
        try:
            mode = {'F': 'fullstage', 'X': 'oneshot'}.get(mode, mode)
            if mode not in ('fullstage', 'oneshot') or runtime.shared_mode != mode:
                raise ValueError('requires matching initialized fullstage/oneshot runtime')
            if logical_alignment and mode != 'oneshot':
                raise ValueError('logical alignment requires physical E384 oneshot')
            if len(runtime.layers) != 16 or {e['state'].cap_experts for e in runtime.layers.values()} != {20 if mode == 'fullstage' else 21}:
                raise ValueError('requires all 16 initialized layers with the mode private capacity')
            report['mode'] = mode; torch = runtime.torch
            for name, expected in EXPECTED.items():
                module = importlib.import_module('vllm.model_executor.layers.fused_moe.' + name)
                path = Path(module.__file__); digest = hashlib.sha256(path.read_bytes()).hexdigest()
                report['sources'][name] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
                if digest != expected: raise RuntimeError('unreviewed installed source: ' + name)
            moe = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
            w1, w2 = (tuple(t[320:384] for t in runtime.shared_weights)
                      if mode == 'fullstage' else runtime.shared_weights)
            e = 64 if mode == 'fullstage' else 384; mapped = mode != 'fullstage'
            global_e = 64 if logical_alignment else e
            if (tuple(w1.shape), tuple(w2.shape)) != ((e, 2048, 2048), (e, 2048, 1024)):
                raise ValueError('unexpected model or pool weight geometry')
            if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16 or not w1.is_contiguous() or not w2.is_contiguous():
                raise ValueError('requires contiguous BF16 pool weights')
            device = w1.device; report['before'] = snapshot(runtime); report['memory_before'] = memory(torch, device)
            configs = [moe.try_get_optimal_moe_config(w1.shape, w2.shape, 8, None, m) for m in range(1, 161)]
            lengths = [assignment(m, c['BLOCK_SIZE_M'], global_e, mapped)[1] for m, c in enumerate(configs, 1)]
            sizes = dict(x=160*2048, activation=160*8*1024, output=160*8*2048,
                         weights=160*8, ids=160*8, sorted=max(lengths), experts=max(lengths), padded=1)
            for name, size in sizes.items():
                dtype = torch.bfloat16 if name in ('x', 'activation', 'output') else torch.float32 if name == 'weights' else torch.int32
                buffers[name] = torch.empty(size, dtype=dtype, device=device)
            report['temporary_buffer_bytes'] = sum(t.numel()*t.element_size() for t in buffers.values())
            report['memory_with_buffers'] = memory(torch, device)
            kernel = moe.fused_moe_kernel; original = kernel.run
            had_run = 'run' in vars(kernel); own_run = vars(kernel).get('run')
            current = None

            def compile_only(*args, **kwargs):
                current['interceptions'] += 1
                if current['interceptions'] != 1: raise RuntimeError('expected one native launch per helper call')
                kwargs['warmup'] = True
                compiled = original(*args, **kwargs)
                if compiled is None or not hasattr(compiled, '_init_handles'):
                    raise RuntimeError('compile did not return a synchronous CompiledKernel')
                current.update(cache_key=compiled.hash, signature=plain(compiled.src.signature),
                    constexprs=plain(compiled.src.constants), attrs=plain(compiled.src.attrs),
                    handles_initialized_before=compiled.module is not None)
                compiled._init_handles()
                current['handles_initialized_after'] = compiled.module is not None
                if compiled.module is None: raise RuntimeError('driver handles were not initialized')
                return compiled

            kernel.run = compile_only; patched = True
            for m, config in enumerate(configs, 1):
                block = config['BLOCK_SIZE_M']; naive, length = assignment(m, block, global_e, mapped)
                sorted_ids = None if naive else buffers['sorted'][:length]
                expert_ids = buffers['ids'][:m*8] if naive else buffers['experts'][:(length+block-1)//block]
                weights = buffers['weights'][:m*8].view(m, 8)
                output = buffers['output'][:m*8*2048].view(m, 8, 2048)
                for gemm in (1, 2):
                    current = dict(m=m, gemm=gemm, local_experts=e, global_experts=global_e, mapped=mapped,
                        naive=naive, sorted_ids_length=None if naive else length, config=dict(config),
                        interceptions=0, start_perf_ns=time.perf_counter_ns(), status='STARTED')
                    report['calls'].append(current)
                    try:
                        a = (buffers['x'][:m*2048].view(m, 2048) if gemm == 1
                             else buffers['activation'][:m*8*1024].view(m*8, 1024))
                        moe.invoke_fused_moe_triton_kernel(a, w1 if gemm == 1 else w2, output,
                            None, None, weights, sorted_ids, expert_ids, buffers['padded'], gemm == 2,
                            8 if gemm == 1 else 1, config, moe.tl.bfloat16, False, False, False, False, False)
                        if current['interceptions'] != 1: raise RuntimeError('native helper did not enter intercepted kernel')
                        current['status'] = 'COMPLETE'
                    except BaseException:
                        current.update(status='FAILED', error=traceback.format_exc()); raise
                    finally: current['end_perf_ns'] = time.perf_counter_ns()
            report['status'] = 'COMPLETE_COMPILE_AND_LOAD_ONLY'
        except BaseException:
            report.update(status='FAILED', error=traceback.format_exc())
            raise
        finally:
            if patched:
                if had_run: kernel.run = own_run
                else: del kernel.run
            a = output = weights = sorted_ids = expert_ids = None; buffers.clear()
            try:
                if 'before' in report:
                    report['after'] = snapshot(runtime); report['state_equal'] = report['before'] == report['after']
                    report['memory_after_release'] = memory(runtime.torch, device)
                    if not report['state_equal']: raise RuntimeError('pager CPU state changed during precompile')
            except BaseException:
                report.update(status='FAILED', state_check_error=traceback.format_exc())
                if 'error' not in report: raise
            finally:
                report.update(actual_invocations=len(report['calls']),
                    unique_cache_keys=sorted({c['cache_key'] for c in report['calls'] if 'cache_key' in c}),
                    end_perf_ns=time.perf_counter_ns())
                report['setup_wall_s'] = (report['end_perf_ns']-start)/1e9
                json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return report
