"""Run unchanged F/X with source-declared compile-domain setup and host observation."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import host_probe
from compile_domain import precompile, EXPECTED
from logical_execution import LogicalExecution


def logical_arguments(argv):
    switch = argparse.ArgumentParser(add_help=False)
    switch.add_argument('--logical-alignment', action='store_true')
    return switch.parse_known_args(argv)


def main():
    option, forwarded = logical_arguments(sys.argv[1:])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool-execution', choices=['fullstage', 'oneshot'], required=True)
    args, _ = parser.parse_known_args(forwarded)
    root = Path(__file__).resolve().parent.parent
    destination = args.output / 'jit_probe.json'
    if destination.exists():
        raise FileExistsError(destination)
    logical_path = args.output / 'logical_execution.json'
    if logical_path.exists(): raise FileExistsError(logical_path)
    logical = dict(schema='logical_alignment_execution_v1', logical_alignment=option.logical_alignment, status='STARTED',
        phases={}, installed_sources={}, hooks_restored=False, scope='When enabled, initialization, original request warmup and measurement each use one logical64/False kernel with physical E384 weights. No references, numeric readback or CPU/GC observer. Aggregate interface recording and all wrapper cost retained.',
        identity_encoding='Per phase, trace order; SHA256 of compact JSON [call_id,layer_name,context.step_id,rows] plus newline per call.')
    sys.path.insert(0, str(root / 'source'))
    report = dict(schema='covered_moe_host_probe_v1', logical_alignment=option.logical_alignment, status='STARTED', events=[],
                  started_perf_ns=time.perf_counter_ns(), installed_sources={},
                  observer_sources={name: hashlib.sha256((root / 'instrumentation' / name).read_bytes()).hexdigest()
                                    for name in ['run_covered.py', 'host_probe.py', 'compile_domain.py', 'logical_execution.py']},
                  cache_environment={k: v for k, v in os.environ.items() if k.startswith('TRITON_')},
                  scope='Compiler/cache/load host intervals; actual F/X applies are not assumed to use the U apply hook. '
                        'Coverage is zero compiler_pipeline during measurement, not a steady-state or speedup guarantee.')
    probe = engine_class = original_descriptor = None
    logical_hook = pager = original_install = None
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], *forwarded]
    try:
        if option.logical_alignment and args.pool_execution != 'oneshot':
            raise ValueError('--logical-alignment requires oneshot')
        for name, expected in host_probe.EXPECTED.items():
            package, relative = name.split('/', 1)
            path = Path(importlib.util.find_spec(package).origin).parent / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            report['installed_sources'][name] = actual
            if actual != expected:
                raise RuntimeError('installed source changed: ' + name)
        triton = importlib.import_module('triton')
        compiler = importlib.import_module('triton.compiler.compiler')
        pager = importlib.import_module('wisp_v026_adapter')
        if option.logical_alignment:
            original_install = pager.install
            def install(*a, **kw):
                nonlocal logical_hook
                runtime = original_install(*a, **kw)
                if logical_hook is None:
                    for name, expected in EXPECTED.items():
                        module = importlib.import_module('vllm.model_executor.layers.fused_moe.' + name)
                        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                        logical['installed_sources'][name] = actual
                        if actual != expected: raise RuntimeError('logical installed source changed: ' + name)
                    module = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
                    logical_hook = LogicalExecution(runtime, module, logical)
                return runtime
            pager.install = install
        groups = importlib.import_module('wisp_expert_groups')
        probe = host_probe.Probe(pager)
        probe.install(triton, compiler, groups)
        from vllm.v1.engine.llm_engine import LLMEngine
        engine_class = LLMEngine
        original_descriptor = engine_class.__dict__['from_engine_args']
        original_constructor = engine_class.from_engine_args

        def construct(cls, *a, **kw):
            engine = original_constructor(*a, **kw)
            runtime = pager._runtime
            core = engine.engine_core.engine_core
            model_runner = core.model_executor.driver_worker.worker.model_runner
            def kv_state():
                pool = core.scheduler.kv_cache_manager.block_pool
                return dict(requests=sorted(core.scheduler.requests), free_blocks=pool.get_num_free_blocks(),
                    total_blocks=pool.num_gpu_blocks,
                    storages=[dict(pointer=t.untyped_storage().data_ptr(), bytes=t.untyped_storage().nbytes())
                              for t in model_runner.kv_caches])
            report['kv_before_compile_setup'] = kv_state()
            old_context = runtime.context
            runtime.context = dict(phase='compile_domain_warmup', step_id=None)
            try:
                precompile(runtime, args.pool_execution, args.output / 'compile_domain.json', logical_alignment=option.logical_alignment)
            finally:
                runtime.context = old_context
                report['kv_after_compile_setup'] = kv_state()
            if report['kv_before_compile_setup'] != report['kv_after_compile_setup']:
                raise RuntimeError('KV storage/pool/request state changed during compile setup')
            return engine

        engine_class.from_engine_args = classmethod(construct)
        runpy.run_path(str(root / 'source' / 'run_shared_pool_pager.py'), run_name='__main__')
        report['status'] = 'COMPLETE'
        logical['status'] = 'COMPLETE'
    except BaseException:
        logical.update(status='FAILED', error=traceback.format_exc())
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        sys.argv = saved_argv
        if logical_hook is not None: logical_hook.close()
        else: logical['hooks_restored'] = True
        if original_install is not None:
            pager.install = original_install
            logical['hooks_restored'] &= pager.install is original_install
        if original_descriptor is not None:
            engine_class.from_engine_args = original_descriptor
        if probe is not None:
            probe.close()
            report['events'] = sorted(probe.events, key=lambda r: r['event_id'])
        report['finished_perf_ns'] = time.perf_counter_ns()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        with logical_path.open('x') as stream:
            json.dump(logical, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
