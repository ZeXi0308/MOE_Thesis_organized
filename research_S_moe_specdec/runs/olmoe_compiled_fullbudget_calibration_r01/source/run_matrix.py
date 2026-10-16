"""Resident engine with wave or queued admission; called under the GPU lock.

Wave keeps the historical sum-of-wave-window metrics. Queued submits the entire
population before the first engine step and relies on max_num_seqs for admission.
The cap bounds running requests, not the actual decode batch on mixed/tail steps.
"""
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


def resolve_admission(plan, batches, override=None):
    mode = override or plan.get('admission_mode', 'wave')
    if mode not in ('wave', 'queued'):
        raise ValueError('admission_mode must be wave or queued')
    if not batches or any(type(b) is not int or b <= 0 for b in batches):
        raise ValueError('batches must contain positive integers')
    cap = plan.get('max_num_seqs', max(batches))
    if type(cap) is not int or cap < max(batches):
        raise ValueError('max_num_seqs must be an integer at least max(batches)')
    if mode == 'queued' and (len(plan['batches']) != 1 or len(batches) != 1
                             or batches != plan['batches'] or cap != batches[0]):
        raise ValueError('queued admission requires one unchanged plan batch and max_num_seqs equal to that batch')
    if plan.get('warmup_full_population',False) and (len(batches) != 1 or cap != batches[0]):
        raise ValueError('full-population shape warmup requires one batch equal to max_num_seqs')
    population = plan.get('measured_requests',64)
    if type(population) is not int or population <= 0:
        raise ValueError('measured_requests must be a positive integer')
    return mode, cap


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
    p.add_argument('--admission-mode', choices=['wave', 'queued'])
    p.add_argument('--native-table', help='JSON list of [min_batch,max_batch,gamma] entries')
    args = p.parse_args()
    if os.environ.get('MOE_S_GPU_LOCKED') != '1':
        raise RuntimeError('run through run_campaign.py so the shared GPU lock is held')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.config).read_text())
    dataset = json.loads(Path(plan['workload']).read_text())
    batches = args.batches or plan['batches']
    admission_mode, engine_max_num_seqs = resolve_admission(plan, batches, args.admission_mode)
    skip_trace = args.skip_trace or plan.get('skip_trace', False)
    repeats = args.repeats or plan['repeats']
    output_tokens = args.output_tokens or plan['output_tokens']
    budget = args.proposal_budget if args.proposal_budget is not None else args.gamma
    token_boundary = plan.get('semantic_token_boundary')
    prompt_boundary_count = sum(token >= token_boundary for record in dataset['records']
        for token in record['prompt_token_ids']) if token_boundary is not None else None
    dump(out/'configuration.json', {'plan':plan,'args':vars(args),
        'admission_mode':admission_mode,'engine_max_num_seqs':engine_max_num_seqs,
        'workload_sha256':hashlib.sha256(Path(plan['workload']).read_bytes()).hexdigest(),
        'semantic_token_boundary':token_boundary,'all_workload_prompt_tokens_at_or_above_boundary':prompt_boundary_count})
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_V2_MODEL_RUNNER='0', WISP_PLUGIN_DISABLE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_BATCH_INVARIANT='0', TOKENIZERS_PARALLELISM='false')
    import torch
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm_s_plugin import install
    from cost_model import AcceptanceEWMA
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
        'disable_above_batch':plan.get('disable_above_batch',8), 'policy':policy,
        'acceptance':policy.acceptance if policy else AcceptanceEWMA(),
        'keep_token_routes':plan.get('keep_token_routes',False),
        'native_route_return':plan.get('native_route_return',False),
        'native_route_backend':plan.get('native_route_backend','gpu'),
        'native_capture_padding':plan.get('native_capture_padding',False),
        'strict':plan.get('enforce_eager',True)})
    kwargs = dict(model=plan['model'],tokenizer=plan['model'],dtype=plan.get('dtype','bfloat16'),seed=20261016,
        max_model_len=plan.get('max_model_len',1024),max_num_seqs=engine_max_num_seqs,max_num_batched_tokens=plan.get('max_num_batched_tokens',16384),
        kv_cache_memory_bytes=plan.get('kv_cache_memory_bytes',6*1024**3),gpu_memory_utilization=.9,
        enable_chunked_prefill=True,enable_prefix_caching=False,async_scheduling=False,
        enable_return_routed_experts=plan.get('native_route_return',False),
        enforce_eager=plan.get('enforce_eager',True),kernel_config=plan.get('kernel_config',{'moe_backend':'triton'}),disable_log_stats=False)
    if plan.get('quantization'):
        kwargs['quantization']=plan['quantization']
    if budget > 0:
        kwargs['speculative_config'] = {'method':plan['draft_source'], 'num_speculative_tokens':budget}
        if plan['draft_source'] in ('ngram','ngram_gpu'):
            kwargs['speculative_config'].update(prompt_lookup_min=2,prompt_lookup_max=5)
        if plan.get('draft_model'):
            kwargs['speculative_config']['model']=plan['draft_model']
        kwargs['speculative_config'].update(plan.get('speculative_overrides',{}))
        if args.native_table:
            kwargs['speculative_config']['num_speculative_tokens_per_batch_size']=json.loads(Path(args.native_table).read_text())
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
            'model_runner_version':'v1_explicit','VLLM_USE_V2_MODEL_RUNNER':os.environ['VLLM_USE_V2_MODEL_RUNNER'],
            'gpu':torch.cuda.get_device_name(0),'load_seconds':time.monotonic()-load_start,
            'allocated_bytes':torch.cuda.memory_allocated(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})

        def episode(label,batch,tokens,offset=0,trace=False,measure=True,save=True,
                    population=None,population_clock=None,admission=None):
            request_count = batch if population is None else population
            records=dataset['records'][offset:offset+request_count]
            if len(records)!=request_count:
                raise ValueError('Insufficient distinct dataset requests')
            runtime.start_episode(label,trace=trace)
            params = SamplingParams(temperature=0,max_tokens=tokens,ignore_eos=not args.natural_stop,
                detokenize=False,output_kind=RequestOutputKind.CUMULATIVE)
            request_rows={}
            steps=[]
            torch.cuda.synchronize()
            t0=time.perf_counter_ns()
            # Subsequent waves share the exact first measurement origin. Their
            # requests are externally ready at zero but wait locally to submit.
            population_clock = {} if population_clock is None else population_clock
            population_t0_ns = population_clock.setdefault('origin_ns',t0)
            if measure:
                meter.start()
            for i,record in enumerate(records):
                rid=f'{label}-{i}'
                request_rows[rid]={'request_id':rid,'source_index':record['source_index'],
                    'prompt_tokens':len(record['prompt_token_ids']),'chunks':[],
                    'output_token_ids':[],'finished':False,
                    'release_elapsed_s':0.0,
                    'submit_elapsed_s':(time.perf_counter_ns()-population_t0_ns)/1e9,
                    'first_output_elapsed_s':None,'last_output_elapsed_s':None,
                    'completion_elapsed_s':None}
                engine.add_request(rid,{'prompt_token_ids':record['prompt_token_ids']},params)
                request_rows[rid]['submit_complete_elapsed_s']=(time.perf_counter_ns()-population_t0_ns)/1e9
            while engine.has_unfinished_requests():
                step0=time.perf_counter_ns()
                outputs=engine.step()
                step1=time.perf_counter_ns()
                steps.append({'index':len(steps),'host_step_ms':(step1-step0)/1e6,
                              'elapsed_s':(step1-t0)/1e9,
                              'population_elapsed_s':(step1-population_t0_ns)/1e9})
                for output in outputs:
                    row=request_rows[output.request_id]
                    item=output.outputs[0]
                    ids=list(item.token_ids)
                    added=len(ids)-len(row['output_token_ids'])
                    if added:
                        row['chunks'].append({'elapsed_s':(step1-t0)/1e9,'tokens':added})
                        if row['first_output_elapsed_s'] is None:
                            row['first_output_elapsed_s']=(step1-population_t0_ns)/1e9
                        row['last_output_elapsed_s']=(step1-population_t0_ns)/1e9
                    row['output_token_ids']=ids
                    row['finished']=output.finished
                    row['finish_reason']=item.finish_reason
                    row['stop_reason']=item.stop_reason
                    if output.finished:
                        row['completion_elapsed_s']=(step1-population_t0_ns)/1e9
            torch.cuda.synchronize()
            measurement_end_ns=time.perf_counter_ns()
            wall_s=(measurement_end_ns-t0)/1e9
            energy=meter.stop() if measure else None
            # GPU route reductions are synchronized only after the measured window.
            plugin=runtime.end_episode()
            rows=list(request_rows.values())
            ntokens=sum(len(r['output_token_ids']) for r in rows)
            for row in rows:
                chunks=row['chunks']
                n=len(row['output_token_ids'])
                row['effective_tpot_ms']=(chunks[-1]['elapsed_s']-chunks[0]['elapsed_s'])*1000/(n-1) if n>1 else None
                if token_boundary is not None:
                    row['output_tokens_at_or_above_semantic_boundary'] = sum(
                        token >= token_boundary for token in row['output_token_ids'])
            result={'label':label,'batch':batch,'gamma':args.gamma,'proposal_budget':budget,
                'mode':args.mode,'draft_source':plan['draft_source'],'model':plan['model'],
                'measurement_kind':'route_diagnostic' if trace or plan.get('native_route_return',False) else 'natural_generation' if args.natural_stop else 'fixed_length_generation',
                'trace_enabled':trace or plan.get('native_route_return',False),
                'route_histogram_enabled':trace,'native_route_return':plan.get('native_route_return',False),
                'execution_mode':'eager' if plan.get('enforce_eager',True) else 'compiled',
                'admission_mode':admission or admission_mode,'engine_max_num_seqs':engine_max_num_seqs,
                'population':request_count,'external_release':'all population requests ready at elapsed_s=0',
                'request_time_origin':'shared population measurement origin; chunks remain episode-local',
                'population_window_start_elapsed_s':(t0-population_t0_ns)/1e9,
                'population_window_end_elapsed_s':(measurement_end_ns-population_t0_ns)/1e9,
                'wall_s':wall_s,'output_tokens':ntokens,
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

        episode('warmup',max(batches),plan['warmup_tokens'],offset=plan.get('warmup_offset',80),measure=False)
        if plan.get('warmup_each_batch',False):
            # Defaults preserve historical runs. Policy evaluation explicitly
            # uses a disjoint development population to initialize its EWMA.
            # Keep each warmup output as provenance; never subtract it from measurements.
            for batch in batches:
                for offset in range(0,plan.get('measured_requests',64),batch):
                    episode(f'shape-warmup-b{batch}-w{offset//batch}',batch,
                        plan['warmup_tokens'],offset=offset+plan.get('shape_warmup_offset',plan.get('input_offset',0)),measure=False)
        if plan.get('warmup_full_population',False):
            # Both diagnostic arms execute this identical queued shape warmup,
            # including mixed prefill/decode and backfill shapes, before timing.
            for batch in batches:
                episode(f'population-shape-warmup-b{batch}',batch,plan['warmup_tokens'],
                    offset=plan.get('shape_warmup_offset',plan.get('input_offset',0)),population=plan.get('measured_requests',64),
                    measure=False,admission='queued')
        dump(out/'evaluation_initial_state.json',{
            'warmup_offset':plan.get('warmup_offset',80),
            'shape_warmup_offset':plan.get('shape_warmup_offset',plan.get('input_offset',0)),
            'input_offset':plan.get('input_offset',0),
            'acceptance_snapshot_before':runtime.acceptance.snapshot(),
            'acceptance_state_policy':'continuous across warmup, batches, repeats and waves; no reset'})
        for repeat in range(repeats):
            for batch in (batches if repeat%2==0 else list(reversed(batches))):
                label=f'perf-r{repeat}-b{batch}'
                population=plan.get('measured_requests',64)
                if admission_mode == 'wave' and population%batch:
                    raise ValueError('measured_requests must be divisible by every batch')
                population_clock={}
                if admission_mode == 'queued':
                    waves=[episode(f'{label}-q0',batch,output_tokens,
                        offset=plan.get('input_offset',0),population=population,save=False,
                        population_clock=population_clock)]
                else:
                    waves=[episode(f'{label}-w{offset//batch}',batch,output_tokens,
                        offset=offset+plan.get('input_offset',0),save=False,
                        population_clock=population_clock) for offset in range(0,population,batch)]
                wall=sum(w['wall_s'] for w in waves)
                tokens=sum(w['output_tokens'] for w in waves)
                energy=sum(w['energy_j'] for w in waves) if all(w['energy_j'] is not None for w in waves) else None
                result={k:waves[0][k] for k in ['batch','gamma','proposal_budget','mode','draft_source','model','measurement_kind','trace_enabled','execution_mode','native_route_return','admission_mode','engine_max_num_seqs','external_release','request_time_origin']}
                population_wall=waves[-1]['population_window_end_elapsed_s']
                inter_episode_gap=population_wall-wall
                if inter_episode_gap < -1e-9:
                    raise RuntimeError('Population window is shorter than its episode windows')
                result.update(label=label,wall_s=wall,output_tokens=tokens,tokens_per_second=tokens/wall,
                    energy_j=energy,joules_per_token=energy/tokens if energy is not None else None,
                    all_finished=all(w['all_finished'] for w in waves),
                    requests=[r for w in waves for r in w['requests']],episodes=waves,
                    population=population,
                    population_wall_s=population_wall,population_tokens_per_second=tokens/population_wall,
                    population_inter_episode_gap_s=max(0.0,inter_episode_gap),
                    population_inter_episode_gap_boundary='includes meter stop/start, runtime summary, Python handling and next-episode preparation between measured windows; no subtraction from population_wall_s',
                    population_timing_boundary='first submission-window origin through final GPU synchronization; includes local waits and inter-wave processing; excludes final post-window processing',
                    wave_arrivals='all requests submitted before first step; scheduler fills freed slots up to engine cap' if admission_mode=='queued' else 'synchronous within wave; next wave after full drain',
                    timing_boundary='single complete population window including all submissions, queue wait, prefill and decode' if admission_mode=='queued' else 'sum of wave windows including request submission, prefill and decode; excludes post-window route/log processing')
                dump(out/f'{label}.json',result)
                print(json.dumps({k:result[k] for k in ['label','batch','gamma','wall_s','tokens_per_second','joules_per_token','all_finished']}),flush=True)
        if not skip_trace:
            for batch in batches:
                population = plan.get('trace_requests', batch)
                if admission_mode == 'wave' and population % batch:
                    raise ValueError('trace_requests must be divisible by every batch')
                if population == batch or admission_mode == 'queued':
                    episode(f'routes-b{batch}',batch,plan['trace_tokens'],offset=plan.get('input_offset',0),trace=True,population=population)
                else:
                    population_clock={}
                    waves = [episode(f'routes-b{batch}-w{offset//batch}',batch,plan['trace_tokens'],
                        offset=offset+plan.get('input_offset',0),trace=True,save=False,
                        population_clock=population_clock)
                        for offset in range(0,population,batch)]
                    result = {key:waves[0][key] for key in ['batch','gamma','proposal_budget','mode',
                        'draft_source','model','measurement_kind','trace_enabled','execution_mode','native_route_return','admission_mode','engine_max_num_seqs']}
                    result.update(label=f'routes-b{batch}',population=population,episodes=waves,
                        requests=[row for wave in waves for row in wave['requests']],
                        all_finished=all(wave['all_finished'] for wave in waves),
                        timing_boundary='route diagnostic; every wave instrumented; excluded from performance')
                    dump(out/f'routes-b{batch}.json',result)
                    print(json.dumps({'label':result['label'],'population':population,'trace_waves':len(waves),
                        'route_events':sum(len(wave['plugin']['routes']) for wave in waves)}),flush=True)
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
