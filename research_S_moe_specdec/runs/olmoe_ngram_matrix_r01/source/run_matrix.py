"""One resident engine, fixed gamma, multiple batches; called under GPU lock."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gamma', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', default='native', choices=['native','static','adaptive','dense_heuristic'])
    p.add_argument('--calibration')
    p.add_argument('--proposal-budget', type=int)
    p.add_argument('--natural-stop', action='store_true')
    p.add_argument('--batches', type=int, nargs='+')
    p.add_argument('--output-tokens', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--skip-trace', action='store_true')
    args = p.parse_args()
    if os.environ.get('MOE_S_GPU_LOCKED') != '1':
        raise RuntimeError('run through run_campaign.py so the shared GPU lock is held')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.config).read_text())
    dataset = json.loads(Path(plan['workload']).read_text())
    batches = args.batches or plan['batches']
    repeats = args.repeats or plan['repeats']
    output_tokens = args.output_tokens or plan['output_tokens']
    budget = args.proposal_budget if args.proposal_budget is not None else args.gamma
    dump(out/'configuration.json', {'plan':plan,'args':vars(args),
        'workload_sha256':hashlib.sha256(Path(plan['workload']).read_bytes()).hexdigest()})
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', WISP_PLUGIN_DISABLE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_BATCH_INVARIANT='0', TOKENIZERS_PARALLELISM='false')
    import torch
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm_s_plugin import install
    from power import PowerSampler
    if vllm.__version__ != '0.26.0':
        raise RuntimeError(f'Expected vLLM 0.26.0, found {vllm.__version__}')
    torch.set_num_threads(8)
    policy = None
    if args.mode == 'adaptive':
        from cost_model import AdaptiveGammaPolicy, CalibrationTable, AcceptanceEWMA
        if not args.calibration:
            raise ValueError('adaptive mode requires measured calibration table')
        policy = AdaptiveGammaPolicy(CalibrationTable.from_json(args.calibration), AcceptanceEWMA())
    runtime = install({'mode':args.mode, 'max_gamma':budget, 'static_gamma':args.gamma,
        'disable_above_batch':8, 'policy':policy})
    kwargs = dict(model=plan['model'],tokenizer=plan['model'],dtype='bfloat16',seed=20261016,
        max_model_len=1024,max_num_seqs=max(batches),max_num_batched_tokens=16384,
        kv_cache_memory_bytes=6*1024**3,gpu_memory_utilization=.9,
        enable_chunked_prefill=True,enable_prefix_caching=False,async_scheduling=False,
        enforce_eager=True,kernel_config={'moe_backend':'triton'},disable_log_stats=False)
    if budget > 0:
        kwargs['speculative_config'] = {'method':plan['draft_source'], 'num_speculative_tokens':budget,
            'prompt_lookup_min':2,'prompt_lookup_max':5}
    dump(out/'engine_args.json',kwargs)
    load_start = time.monotonic()
    engine = None
    meter = None
    try:
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs),enable_multiprocessing=False)
        runtime.attach_engine(engine)
        meter = PowerSampler(interval_s=.05)
        dump(out/'environment.json',{'python':sys.version,'vllm':vllm.__version__,
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'load_seconds':time.monotonic()-load_start,
            'allocated_bytes':torch.cuda.memory_allocated(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})

        def episode(label,batch,tokens,offset=0,trace=False,measure=True,save=True):
            records=dataset['records'][offset:offset+batch]
            if len(records)!=batch:
                raise ValueError('Insufficient distinct dataset requests')
            runtime.start_episode(label,trace=trace)
            params = SamplingParams(temperature=0,max_tokens=tokens,ignore_eos=not args.natural_stop,
                detokenize=False,output_kind=RequestOutputKind.CUMULATIVE)
            request_rows={}
            steps=[]
            torch.cuda.synchronize()
            t0=time.perf_counter_ns()
            if measure:
                meter.start()
            for i,record in enumerate(records):
                rid=f'{label}-{i}'
                request_rows[rid]={'request_id':rid,'source_index':record['source_index'],
                    'prompt_tokens':len(record['prompt_token_ids']),'chunks':[],
                    'output_token_ids':[],'finished':False}
                engine.add_request(rid,{'prompt_token_ids':record['prompt_token_ids']},params)
            while engine.has_unfinished_requests():
                step0=time.perf_counter_ns()
                outputs=engine.step()
                step1=time.perf_counter_ns()
                steps.append({'index':len(steps),'host_step_ms':(step1-step0)/1e6,
                              'elapsed_s':(step1-t0)/1e9})
                for output in outputs:
                    row=request_rows[output.request_id]
                    item=output.outputs[0]
                    ids=list(item.token_ids)
                    added=len(ids)-len(row['output_token_ids'])
                    if added:
                        row['chunks'].append({'elapsed_s':(step1-t0)/1e9,'tokens':added})
                    row['output_token_ids']=ids
                    row['finished']=output.finished
                    row['finish_reason']=item.finish_reason
                    row['stop_reason']=item.stop_reason
            torch.cuda.synchronize()
            wall_s=(time.perf_counter_ns()-t0)/1e9
            energy=meter.stop() if measure else None
            # GPU route reductions are synchronized only after the measured window.
            plugin=runtime.end_episode()
            rows=list(request_rows.values())
            ntokens=sum(len(r['output_token_ids']) for r in rows)
            for row in rows:
                chunks=row['chunks']
                n=len(row['output_token_ids'])
                row['effective_tpot_ms']=(chunks[-1]['elapsed_s']-chunks[0]['elapsed_s'])*1000/(n-1) if n>1 else None
            result={'label':label,'batch':batch,'gamma':args.gamma,'proposal_budget':budget,
                'mode':args.mode,'draft_source':plan['draft_source'],'model':plan['model'],
                'measurement_kind':'route_diagnostic' if trace else 'natural_generation' if args.natural_stop else 'fixed_length_generation',
                'trace_enabled':trace,'wall_s':wall_s,'output_tokens':ntokens,
                'tokens_per_second':ntokens/wall_s,'energy_j':energy['energy_j'] if energy else None,
                'joules_per_token':energy['energy_j']/ntokens if energy and energy['energy_j'] is not None and ntokens else None,
                'requests':rows,'host_steps':steps,'plugin':plugin,'energy':energy,
                'all_finished':all(r['finished'] for r in rows)}
            if save:
                dump(out/f'{label}.json',result)
                print(json.dumps({k:result[k] for k in ['label','batch','gamma','wall_s','tokens_per_second','joules_per_token','all_finished']}),flush=True)
            if len(plugin['steps']) != len(steps):
                dump(out/f'{label}-incomplete.json',result)
                raise RuntimeError('Not every engine step reached the scheduler observer')
            if trace and not plugin['summary']['trace_complete']:
                raise RuntimeError('Route diagnostics incomplete: '+repr(plugin['summary']['route_errors']))
            return result

        episode('warmup',max(batches),plan['warmup_tokens'],offset=80,measure=False)
        for repeat in range(repeats):
            for batch in (batches if repeat%2==0 else list(reversed(batches))):
                label=f'perf-r{repeat}-b{batch}'
                population=plan.get('measured_requests',64)
                if population%batch:
                    raise ValueError('measured_requests must be divisible by every batch')
                waves=[episode(f'{label}-w{offset//batch}',batch,output_tokens,offset=offset,save=False)
                       for offset in range(0,population,batch)]
                wall=sum(w['wall_s'] for w in waves)
                tokens=sum(w['output_tokens'] for w in waves)
                energy=sum(w['energy_j'] for w in waves) if all(w['energy_j'] is not None for w in waves) else None
                result={k:waves[0][k] for k in ['batch','gamma','proposal_budget','mode','draft_source','model','measurement_kind','trace_enabled']}
                result.update(label=label,wall_s=wall,output_tokens=tokens,tokens_per_second=tokens/wall,
                    energy_j=energy,joules_per_token=energy/tokens if energy is not None else None,
                    all_finished=all(w['all_finished'] for w in waves),
                    requests=[r for w in waves for r in w['requests']],episodes=waves,
                    population=population,wave_arrivals='synchronous within wave; next wave after full drain',
                    timing_boundary='sum of wave windows including request submission, prefill and decode; excludes post-window route/log processing')
                dump(out/f'{label}.json',result)
                print(json.dumps({k:result[k] for k in ['label','batch','gamma','wall_s','tokens_per_second','joules_per_token','all_finished']}),flush=True)
        if not args.skip_trace:
            for batch in batches:
                episode(f'routes-b{batch}',batch,plan['trace_tokens'],trace=True)
        dump(out/'status.json',{'status':'complete','end_unix_s':time.time()})
    except BaseException as exc:
        dump(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if meter:
            meter.close()
        if engine:
            engine.engine_core.shutdown()


if __name__=='__main__':
    main()
