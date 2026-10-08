#!/usr/bin/env python3
"""Serial native cell with normally profiled or explicitly pinned GPU KV."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

RUNTIME_SOURCES = ['v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py',
    'v1/core/block_pool.py', 'v1/worker/gpu_model_runner.py', 'v1/core/kv_cache_coordinator.py',
    'v1/core/single_type_kv_cache_manager.py', 'v1/core/kv_cache_utils.py', 'config/model.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(folder):
    config, workload = read(folder/'config.json'), read(folder/'workload.json')
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config['workload_sha256']:
        raise ValueError('workload identity changed')
    rows, prompts = workload['source_requests'], workload['actual_prompt_token_ids']
    if len(rows) != len(prompts) or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('request identities do not align')
    for row, ids in zip(rows, prompts):
        if not ids or hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != row['prompt_token_ids_sha256']:
            raise ValueError('prompt token identity mismatch')
    return config, workload


def validated_output_overrides(config, workload):
    """Validate assigned measurement budgets without changing prompt or arrival data."""
    maximum = config.get('output_tokens', 1024)
    if type(maximum) is not int or not 2 <= maximum <= 1024:
        raise ValueError('Global output budget must be an integer in 2..1024')
    overrides = config.get('output_tokens_by_request', {})
    rows, prompts = workload['source_requests'], workload['actual_prompt_token_ids']
    request_ids = [row['request_id'] for row in rows]
    if (len(rows) != len(prompts) or len(set(request_ids)) != len(rows)
            or not isinstance(overrides, dict) or set(overrides) - set(request_ids)):
        raise ValueError('Output overrides must map known, aligned request identities')
    if any(type(value) is not int or not 2 <= value <= maximum for value in overrides.values()):
        raise ValueError('Per-request output budget must be an integer in 2..global budget')
    if (config.get('max_model_len', 4096) != 4096
            or any(not prompt or len(prompt) + overrides.get(rid, maximum) > 4096
                   for rid, prompt in zip(request_ids, prompts))):
        raise ValueError('Prompt plus assigned output budget exceeds native 4096-token context')
    return dict(overrides)


def gpu_state():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], text=True).strip()
    if any(row and int(row[0]) != os.getpid() for row in csv.reader(processes.splitlines())):
        raise RuntimeError(f'GPU has another compute process: {processes!r}')
    return dict(compute_processes=processes, device=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm',
        '--format=csv,noheader'], text=True).strip())


def eos_metadata(engine):
    """Read the installed ModelConfig API; never modify generation defaults."""
    try:
        model = engine.vllm_config.model_config
        generation = model.try_get_generation_config()
        return dict(status='READ', hf_eos_token_id=model.hf_config.to_dict().get('eos_token_id'),
            generation_config_source=model.generation_config,
            generation_config=generation, generation_overrides=model.override_generation_config,
            effective_sampling_defaults=model.get_diff_sampling_param(),
            scope='ModelConfig metadata; actual request finish/stop reasons come from capture.')
    except Exception as error:
        return dict(status='UNAVAILABLE', error=f'{type(error).__name__}: {error}')



def install_policy(scheduler, vllm_config, block_size, variant, diagnostic, cooldown, installer,
                   commit_recheck=False, fit_first_resume=False, capacity_victim=False, recovery_min_outputs=1, yield_to_ready_head=False, spare_followup=False, ordinary_backfill=False, oldest_admission_mode=None, oldest_repeat=False):
    if variant != 'native_full_native':
        return installer(scheduler, vllm_config=vllm_config, block_size=block_size,
            population_mode='open', save=True, global_cooldown_steps=cooldown,
            diagnostic=diagnostic, store_scope='native_full' if variant=='native_full_ordinary_only' else 'selected',
            allow_forced_rotations=(oldest_admission_mode=='queue_fund'
                                    or variant!='native_full_ordinary_only'),
            commit_recheck=commit_recheck, fit_first_resume=fit_first_resume, capacity_victim=capacity_victim,
            recovery_min_outputs=recovery_min_outputs, yield_to_ready_head=yield_to_ready_head,
            spare_followup=spare_followup, ordinary_backfill=ordinary_backfill,
            oldest_admission_mode=oldest_admission_mode,oldest_repeat=oldest_repeat)
    cs=scheduler.connector.connector_scheduler
    hooks=('_rotation_begin','_rotation_hold','_rotation_target','_rotation_forced_count')
    if ('schedule' in vars(scheduler) or '_calc_num_offloadable_tokens' in vars(cs)
            or any(h in vars(scheduler) for h in hooks) or cs.config.offload_prompt_only):
        raise RuntimeError('Native reference requires unmodified native schedule/full saving')
    data=dict(status='NOT_APPLICABLE', rotation_status='NOT_APPLICABLE',
        save=True, store_scope='native_full', native_calc_overridden=False,
        allow_forced_rotations=False,
        scheduler_schedule_overridden=False, diagnostic=False, rotation_config=None,
        applied_rotations=None, direct_commits=None, commit_recheck=False,
        events=[], eligibility_snapshots=[], gate_observations=[],
        selector_decisions=None, scope='No rotation adapter installed. Empty policy logs are not event measurements; native save/load job counts remain NOT_MEASURED.')
    return data, lambda:data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--warmup-inputs', type=Path, required=True)
    parser.add_argument('--variant', choices=['current','eager','native_full_native','native_full_ordinary_only'], required=True)
    parser.add_argument('--measurement-mode', choices=['performance','diagnostic'], required=True)
    parser.add_argument('--commit-recheck', action='store_true')
    parser.add_argument('--fit-first-resume', action='store_true')
    parser.add_argument('--capacity-victim', action='store_true')
    parser.add_argument('--yield-to-ready-head', action='store_true')
    parser.add_argument('--spare-followup', action='store_true')
    parser.add_argument('--ordinary-backfill', action='store_true')
    parser.add_argument('--recovery-min-outputs', type=int, choices=(1,10), default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.commit_recheck and args.variant != 'eager':
        parser.error('--commit-recheck is qualified only for the eager selected-save arm')
    if args.fit_first_resume and (args.variant != 'eager' or args.commit_recheck):
        parser.error('fit-first requires eager with commit-recheck off')
    if args.capacity_victim and (args.variant != 'eager' or args.commit_recheck or args.fit_first_resume):
        parser.error('capacity-victim requires eager with other candidate flags off')
    if args.recovery_min_outputs != 1 and not args.capacity_victim:
        parser.error('extended recovery protection requires capacity-victim')
    if args.spare_followup and (args.recovery_min_outputs!=1 or not args.capacity_victim or args.yield_to_ready_head):
        parser.error('spare-followup requires capacity-only Q1')
    if args.ordinary_backfill and (args.spare_followup or args.recovery_min_outputs!=1
            or args.yield_to_ready_head or args.fit_first_resume or args.commit_recheck
            or not ((args.variant=='eager' and args.capacity_victim)
                    or (args.variant=='native_full_ordinary_only' and not args.capacity_victim))):
        parser.error('ordinary-backfill requires eager capacity Q1 or native-full ordinary-only')
    if args.variant=='native_full_ordinary_only' and (not args.ordinary_backfill
            or args.capacity_victim or args.spare_followup or args.yield_to_ready_head
            or args.commit_recheck or args.fit_first_resume or args.recovery_min_outputs!=1):
        parser.error('native-full ordinary-only requires only ordinary-backfill Q1')
    if args.yield_to_ready_head and (args.recovery_min_outputs!=10 or not args.capacity_victim):
        parser.error('yield requires capacity-qualified Q10')
    victim_rule=os.environ.get('A_NATIVE_VICTIM_RULE','tail')
    if victim_rule not in ('tail','arrival','service_density','bidkv_score','host_near','equal_held_once','partial_restore_once',
                           'min_held_other','max_held_other'):
        parser.error('Unsupported native victim rule')
    full_running_mode=os.environ.get('A_NATIVE_VICTIM_FULL_RUNNING','off')
    if full_running_mode not in ('off','on'):
        parser.error('A_NATIVE_VICTIM_FULL_RUNNING must be off or on')
    if full_running_mode=='on' and (args.variant!='native_full_ordinary_only'
            or victim_rule!='bidkv_score'):
        parser.error('full-running victim set requires native-full ordinary-only BidKV')
    deferral_mode=os.environ.get('A_NATIVE_CAPACITY_DEFERRAL','off')
    if deferral_mode not in ('off','prefix_work','prefix_finish'):
        parser.error('A_NATIVE_CAPACITY_DEFERRAL must be off, prefix_work, or prefix_finish')
    if deferral_mode!='off' and (args.variant!='native_full_ordinary_only'
            or victim_rule!='tail'
            or full_running_mode!='off'
            or os.environ.get('A_NATIVE_VICTIM_CURRENT_GUARD','off')!='off'
            or os.environ.get('A_SELF_PREEMPT_CONTINUE','off')!='off'):
        parser.error('capacity deferral requires native-full ordinary-only, tail, and other victim probes off')
    if victim_rule in ('min_held_other','max_held_other') and (
            args.variant!='native_full_ordinary_only' or full_running_mode!='off'
            or deferral_mode!='off'
            or os.environ.get('A_NATIVE_VICTIM_CURRENT_GUARD','off')!='off'
            or os.environ.get('A_SELF_PREEMPT_CONTINUE','off')!='off'):
        parser.error('size victim rule requires native-full ordinary-only and other probes off')
    oldest_mode=os.environ.get('A_NATIVE_OLDEST_ADMISSION')
    if oldest_mode not in ('ordinary','native','queue_only','queue_fund'):
        parser.error('A_NATIVE_OLDEST_ADMISSION must be ordinary, native, queue_only, or queue_fund')
    lease_mode=os.environ.get('A_RECOVERY_LEASE_MODE','off')
    if lease_mode not in ('off','adaptive','fixed4','q1'):
        parser.error('A_RECOVERY_LEASE_MODE must be off, adaptive, fixed4, or q1')
    if lease_mode!='off' and oldest_mode!='queue_fund':
        parser.error('recovery lease requires queue_fund')
    ordinary_arm=oldest_mode=='ordinary'
    oldest_repeat=not ordinary_arm
    if os.environ.get('A_NATIVE_OLDEST_REPEAT')!=('0' if ordinary_arm else '1'):
        parser.error('A_NATIVE_OLDEST_REPEAT must be 0 for ordinary and 1 otherwise')
    effective_oldest_mode=None if ordinary_arm else oldest_mode
    if (args.variant!='native_full_ordinary_only' or not args.ordinary_backfill
            or victim_rule not in ('tail','host_near','equal_held_once','partial_restore_once') or full_running_mode!='off' or deferral_mode!='off'
            or os.environ.get('A_NATIVE_VICTIM_CURRENT_GUARD','off')!='off'
            or os.environ.get('A_SELF_PREEMPT_CONTINUE','off')!='off'):
        parser.error('oldest admission requires isolated native victim selection and other probes off')
    effective_ordinary_backfill=ordinary_arm
    cooldown = {'current':20, 'eager':0, 'native_full_native':None,
                'native_full_ordinary_only':0}[args.variant]
    rotation_enabled = args.variant != 'native_full_native'
    selected_store = args.variant in ('current','eager')
    diagnostic = args.measurement_mode == 'diagnostic'
    if diagnostic and args.variant != 'eager':
        parser.error('Only the new eager/open combination has a diagnostic cell')
    root, out, engine = Path(__file__).resolve().parent, args.output_dir, None
    timing = dict(process_start_unix_s=time.time(), process_start_perf_s=time.perf_counter())
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'status.json', dict(status='INITIALIZING'))
    (out/'commands.txt').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')
    try:
        from metrics import summarize_episode_requests
        from native_capture import capture_episode, set_empty_admission_cap
        from request_measurement import measure_episode
        from memory_telemetry import capture_with_memory, memory_snapshot
        from safe_static import qualify_safe_cap
        from absence_rotation import RotationConfig
        from staged_store_rotation import install as install_rotation
        from native_offload_observer import drain as drain_offload, install as observe_offload
        from native_host_snapshot import snapshot as host_snapshot
        input_case = os.environ.get('A_INPUT_CASE')
        if input_case not in (None, 'low', 'knee', 'high'):
            raise ValueError('A_INPUT_CASE must be low, knee, high, or unset')
        input_dir = args.inputs / f'pro_{input_case}' if input_case else args.inputs
        config, workload = load_inputs(input_dir)
        prefix_caching = config.get('enable_prefix_caching', False)
        if type(prefix_caching) is not bool:
            raise ValueError('enable_prefix_caching must be an explicit boolean')
        output_overrides = validated_output_overrides(config, workload)
        if output_overrides and diagnostic:
            raise ValueError('Per-request budgets require performance measure_episode, not native_capture')
        lengths = [len(ids) for ids in workload['actual_prompt_token_ids']]
        max_num_seqs = int(os.environ.get('A_MAX_NUM_SEQS', config.get('engine_max_num_seqs', 384)))
        max_seconds = float(os.environ.get('A_MAX_SECONDS', config.get('max_seconds', 600)))
        output_tokens = config.get('output_tokens', 1024)
        profile_flag = os.environ.get('A_PROFILE_ONLY', '0')
        if profile_flag not in ('0', '1'):
            raise ValueError('A_PROFILE_ONLY must be 0 or 1')
        profile_only = profile_flag == '1'
        gpu_kv_value = os.environ.get('A_GPU_KV_BYTES')
        gpu_kv_bytes = int(gpu_kv_value) if gpu_kv_value is not None else None
        if profile_only and gpu_kv_bytes is not None:
            raise ValueError('Profile must omit A_GPU_KV_BYTES')
        if not profile_only and (gpu_kv_bytes is None or gpu_kv_bytes <= 0):
            raise ValueError('Measurement requires positive A_GPU_KV_BYTES from normal profile')
        if (not lengths or type(output_tokens) is not int or output_tokens < 2
                or any(n < 1 for n in lengths)
                or config.get('max_model_len', 4096) != 4096
                or not 1 <= max_num_seqs <= 1024
                or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('Invalid request, sequence, time, or native 4096-token context bound')
        arrivals = workload['arrival_traces_s']['steady']
        if (len(arrivals) != len(lengths)
                or any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in arrivals)
                or max(arrivals) >= max_seconds):
            raise ValueError('Arrival sequence must fit the complete measurement horizon')
        config.update(requests=len(lengths), prompt_tokens=max(lengths), output_tokens=output_tokens, output_mode='eos',
            output_tokens_by_request=output_overrides, cap=max_num_seqs, engine_max_num_seqs=max_num_seqs,
            max_model_len=4096, policy='static', max_seconds=max_seconds,
            variant=args.variant, commit_recheck=args.commit_recheck, fit_first_resume=args.fit_first_resume, capacity_victim=args.capacity_victim, recovery_min_outputs=args.recovery_min_outputs,
            global_cooldown_steps=cooldown, population_mode='open' if rotation_enabled else 'native', preemption_mode='native_recompute_or_external_kv',
            measurement_mode='diagnostic' if diagnostic else 'performance_sparse_preemptions',
            selective_save='on', store_scope='selected' if selected_store else 'native_full', offload_gib=16,
            ignore_eos=False, min_tokens=0,
            fixed_kv_cache_memory_bytes=gpu_kv_bytes, gpu_memory_utilization=.90, reservation_policy='full',
            profile_only=profile_only, input_case=input_case, resolved_input_dir=str(input_dir),
            rotation_config=vars(RotationConfig(min_steps_between_swaps=cooldown)) if rotation_enabled else None,
            rotation_victim_order='most_output' if selected_store else None,
            prompt_tokens_role='maximum observed input length; output_tokens is requested cap',
            evidence_ceiling='NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT',
            metric_role='Native combination qualification; no performance contrast' if diagnostic else
                'natural complete-service tradeoff; same sparse preemption recording included in all timing arms')
        config['recovery_lease_mode']=lease_mode
        config['funding_victim_rule']=os.environ.get('A_FUNDING_VICTIM_RULE','tail')
        config['recovery_lease_config']=dict(max_quantum=16,overhead_fraction=.5,interruption_floor_s=1.)
        config['recovery_lease_cost_semantics']='past host first-output interval minus median pure-decode begin interval; no added overlapping DMA'
        config['yield_to_ready_head']=args.yield_to_ready_head
        config['spare_followup']=args.spare_followup
        config['ordinary_backfill_cli_requested']=args.ordinary_backfill
        config['ordinary_backfill']=effective_ordinary_backfill
        config['oldest_admission_cli_requested']=oldest_mode
        config['oldest_admission_mode']=effective_oldest_mode
        config['oldest_repeat']=oldest_repeat
        config['last_output_age_semantics']=('first begin-step observation of increased real output count; threshold=1.0s'
            if oldest_repeat else 'NOT_APPLICABLE_TO_ORDINARY_BACKFILL')
        config['allow_forced_rotations']=oldest_mode=='queue_fund'
        config['native_victim_full_running']=full_running_mode=='on'
        config['capacity_deferral_mode']=deferral_mode
        config['native_victim_rule']=victim_rule
        dump(out/'config.json', config)
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0', VLLM_USE_SIMPLE_KV_OFFLOAD='0')
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        if vllm.__version__ != '0.26.0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('requires existing vLLM0.26 single-GPU environment')
        runtime_hashes = {name: sha(Path(vllm.__file__).parent/name) for name in RUNTIME_SOURCES}
        dump(out/'environment.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version('transformers'), gpu_before=before,
            cpu_threads=torch.get_num_threads(), vllm_source_sha256=runtime_hashes,
            source_sha256={name: sha(root/name) for name in ['run_recovery_cadence.py', 'request_measurement.py', 'native_capture.py',
                'memory_telemetry.py', 'metrics.py', 'safe_static.py', 'rotation_native.py', 'absence_rotation.py',
                'staged_store_rotation.py', 'staged_save_contract.py', 'recovery_lease.py', 'funding_victim.py', 'native_offload_observer.py',
                'native_host_snapshot.py', 'native_store_delta.py', 'native_full_store_evidence.py']}))
        if runtime_hashes['v1/core/sched/scheduler.py'] != '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941' or runtime_hashes['v1/core/kv_cache_manager.py'] != '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf':
            raise RuntimeError('pinned runtime source changed')
        model = config['model']
        kwargs = dict(model=model['id'], revision=model['revision'], tokenizer_revision=model['tokenizer_revision'],
            dtype='bfloat16', seed=config['seed'], max_model_len=4096, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=1024, gpu_memory_utilization=.90, enable_chunked_prefill=True,
            enable_prefix_caching=prefix_caching, scheduling_policy='fcfs', async_scheduling=False,
            scheduler_reserve_full_isl=True, stream_interval=1,
            enforce_eager=False, enable_return_routed_experts=False)
        if gpu_kv_bytes is not None:
            kwargs['kv_cache_memory_bytes'] = gpu_kv_bytes
        kwargs.update(kv_offloading_size=16, kv_offloading_backend='native',
            kv_transfer_config={'kv_connector_extra_config': {'offload_prompt_only': False}})
        dump(out/'engine_args.json', kwargs)
        timing['engine_init_start_perf_s']=time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing['engine_init_end_perf_s']=time.perf_counter()
        dump(out/'resolved-eos.json', eos_metadata(engine))
        host_init=host_snapshot(engine)
        dump(out/'host-after-init.json', host_init)
        if ((host_init.get('cpu_kv') or {}).get('unique_storage_bytes') != 16*1024**3
                or (host_init.get('manager') or {}).get('capacity_blocks') != 8192):
            raise RuntimeError('Actual unique host KV allocation/capacity differs from16GiB/8192blocks')
        memory = memory_snapshot(engine, torch)
        dump(out/'memory-after-init.json', memory)
        qualification = qualify_safe_cap(engine, config)
        qualification['observed_scheduler_reserve_full_isl'] = engine.engine_core.engine_core.scheduler.scheduler_reserve_full_isl
        dump(out/'safe-cap-qualification.json', qualification)
        if (qualification['status'] != 'QUALIFIED'
                or not qualification['observed_scheduler_reserve_full_isl']):
            raise RuntimeError('actual KV pool/layout or initial-history reservation failed qualification')
        total_blocks = qualification['total_blocks']
        actual_kv_bytes = memory['kv_storage_bytes']
        if actual_kv_bytes <= 0 or actual_kv_bytes % total_blocks:
            raise RuntimeError('Actual KV allocation does not divide into physical pages')
        page_bytes = actual_kv_bytes // total_blocks
        if page_bytes != host_init['cpu_kv']['bytes_per_host_block']:
            raise RuntimeError('GPU and one-page host KV layouts differ')
        if gpu_kv_bytes is not None and actual_kv_bytes != gpu_kv_bytes:
            raise RuntimeError('Actual KV allocation differs from the exact profiled pin')
        capacity = dict(status='PROFILE_COMPLETE' if profile_only else 'PIN_VERIFIED',
            block_size_tokens=qualification['block_size'], total_blocks=total_blocks,
            usable_blocks=qualification['usable_blocks'], null_blocks=total_blocks-qualification['usable_blocks'],
            kv_page_bytes=page_bytes, actual_kv_storage_bytes=actual_kv_bytes,
            pin_kv_cache_memory_bytes=actual_kv_bytes,
            host_kv_bytes=host_init['cpu_kv']['unique_storage_bytes'],
            host_capacity_blocks=host_init['manager']['capacity_blocks'],
            max_num_seqs=max_num_seqs, max_model_len=4096, max_num_batched_tokens=1024,
            gpu_memory_utilization=.90, profile_only=profile_only,
            semantics='Physical allocation includes the null page; pin this exact byte budget in all policy cells.')
        dump(out/'normal_capacity_profile.json', capacity)
        if profile_only:
            dump(out/'status.json', dict(status='PROFILE_COMPLETE', error=None,
                requests_completed=0, measurement_status='UNRUN', warmup_status='UNRUN'))
            return
        warmup_root = args.inputs if input_case else (args.inputs.parent if args.inputs.name.startswith('pro_') else args.inputs)
        short_work = read(warmup_root/'warmup_short.json')
        warmups = dict(short=(config, short_work), long=load_inputs(args.warmup_inputs/'long'))
        if any(wc['model'] != config['model'] for wc, _ in warmups.values()):
            raise ValueError('warmup and measurement model identities differ')
        timing['warmup_start_perf_s']=time.perf_counter()
        print('PHASE APPLICATION_WARMUP_BEGIN', flush=True)
        for index, (domain, count, cap) in enumerate([
                ('short', min(32, max_num_seqs), min(32, max_num_seqs)),
                ('short', max_num_seqs, max_num_seqs), ('long', 2, 2)]):
            wc, ww = warmups[domain]
            if len(ww['source_requests']) < count or len(ww['actual_prompt_token_ids']) < count:
                raise ValueError('Warmup corpus does not cover configured sequence width')
            work = dict(ww, source_requests=ww['source_requests'][:count],
                actual_prompt_token_ids=ww['actual_prompt_token_ids'][:count], arrival_traces_s={'steady': [0.] * count})
            warm_outputs = (16 if count <= 32 else
                16 + math.ceil(count * max(map(len, work['actual_prompt_token_ids'])) / max(1, 1024-cap)))
            if (len({row['request_id'] for row in work['source_requests']}) != count
                    or any(len(ids)+warm_outputs > 4096 for ids in work['actual_prompt_token_ids'])):
                raise ValueError('Warmup identities or native context bound differ')
            warm_config = dict(wc, cap=cap, requests=count, output_tokens=warm_outputs,
                output_tokens_by_request={}, ignore_eos=True, min_tokens=warm_outputs)
            warm_config.pop('output_mode', None)
            set_empty_admission_cap(engine, cap)
            raw = capture_with_memory(engine, capture_episode, work, warm_config,
                regime='steady', arrival_scale=1., run_id=f'warmup{index}', max_seconds=120)
            raw['warmup_coverage']=dict(requested_width=cap, requested_output_tokens=warm_outputs,
                observed_max_running=max((row['actual_active'] for row in raw['scheduler_steps']), default=0),
                observed_max_pure_decode_width=max((len(row['scheduled']) for row in raw['scheduler_steps']
                    if row['scheduled'] and all(request['decode_tokens']==request['scheduled_tokens']==1
                        for request in row['scheduled'])), default=0),
                semantics='Observed native schedule widths; completion status separately verifies warmup execution.')
            dump(out/f'warmup-{index}.json', raw)
            if raw['status'] != 'COMPLETE':
                raise RuntimeError('warmup incomplete; measurement unrun')
        dump(out/'warmup-offload-drain.json', drain_offload(engine))
        if not engine.reset_prefix_cache(reset_connector=True):
            raise RuntimeError('Failed to clear warmup native connector cache')
        dump(out/'warmup-cache-reset.json', dict(success=True))
        timing['warmup_end_perf_s']=time.perf_counter()
        print('PHASE APPLICATION_WARMUP_END', flush=True)
        set_empty_admission_cap(engine, max_num_seqs)  # Verifies drain before installing either adapter arm.
        before = gpu_state()
        torch.cuda.reset_peak_memory_stats()
        dump(out/'memory-before.json', memory_snapshot(engine, torch))
        dump(out/'host-before.json', host_snapshot(engine))
        scheduler = engine.engine_core.engine_core.scheduler
        dump(out/'resolved-scheduler-config.json', dict(max_num_running_reqs=scheduler.max_num_running_reqs,
            long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold))
        offload_data=dict(status='NOT_MEASURED',diagnostic=False,completed_jobs=None,dispatch=None,transfers=None,
            scope='No lookup/worker/completion observer in either timing arm; saving and transfer qualification are reused from D/E.')
        uninstall_offload=lambda:None
        if diagnostic:
            offload_data, uninstall_offload = observe_offload(scheduler.connector, diagnostic=True,
                host_snapshot=lambda: host_snapshot(engine))
        selective, uninstall = install_policy(scheduler, engine.vllm_config, qualification['block_size'],
            args.variant, diagnostic, cooldown, install_rotation, args.commit_recheck, args.fit_first_resume, args.capacity_victim, args.recovery_min_outputs, args.yield_to_ready_head, args.spare_followup, effective_ordinary_backfill, effective_oldest_mode, oldest_repeat)
        if selective.get('recovery_lease_mode')!=lease_mode:
            raise RuntimeError('Executed recovery lease differs from configuration')
        if selective['store_scope']!=config['store_scope'] or selective['native_calc_overridden']!=selected_store:
            raise RuntimeError('Applied saving/rotation mode differs from the requested arm')
        if selective.get('allow_forced_rotations') != config['allow_forced_rotations']:
            raise RuntimeError('Applied forced rotation permission differs from the requested arm')
        if selective.get('native_victim_full_running_enabled',False) != config['native_victim_full_running']:
            raise RuntimeError('Applied full-running victim set differs from configuration')
        if selective.get('capacity_deferral_mode') != config['capacity_deferral_mode']:
            raise RuntimeError('Applied capacity deferral mode differs from configuration')
        if selective.get('native_victim_rule') != config['native_victim_rule']:
            raise RuntimeError('Applied native victim rule differs from configuration')
        if selective['rotation_config'] != config['rotation_config']:
            raise RuntimeError('Applied rotation configuration differs from requested cadence')
        if selective['commit_recheck'] != config['commit_recheck']:
            raise RuntimeError('Applied commit recheck differs from requested arm')
        if selective.get('fit_first_resume', False) != args.fit_first_resume:
            raise RuntimeError('Applied fit-first flag differs')
        if selective.get('spare_followup', False) != args.spare_followup:
            raise RuntimeError('Executed spare followup differs from configuration')
        if selective.get('ordinary_backfill', False) != effective_ordinary_backfill:
            raise RuntimeError('Executed ordinary backfill differs from configuration')
        if selective.get('oldest_admission_mode') != effective_oldest_mode:
            raise RuntimeError('Applied oldest admission mode differs from configuration')
        if selective.get('oldest_repeat') is not oldest_repeat:
            raise RuntimeError('Applied repeated oldest mode differs from configuration')
        if selective.get('capacity_victim', False) != args.capacity_victim:
            raise RuntimeError('Applied capacity-victim flag differs')
        if selective.get('recovery_min_outputs', 1) != args.recovery_min_outputs:
            raise RuntimeError('Applied recovery minimum differs')
        raw = None
        try:
            try:
                timing['measurement_start_perf_s']=time.perf_counter()
                print('PHASE MEASUREMENT_BEGIN', flush=True)
                if diagnostic:
                    raw = capture_with_memory(engine, capture_episode, workload, config, allow_preemption=True,
                        regime='steady', arrival_scale=1., run_id='measured', max_seconds=max_seconds)
                else:
                    raw = measure_episode(engine, workload, config,
                        regime='steady', arrival_scale=1., run_id='measured', max_seconds=max_seconds, record_preemptions=True)
                timing['measurement_return_perf_s']=time.perf_counter()
                print('PHASE MEASUREMENT_END', flush=True)
                dump(out/'host-request-end.json', host_snapshot(engine))
                dump(out/'post-request-drain.json', drain_offload(engine))
                timing['post_request_drain_end_perf_s']=time.perf_counter()
            finally:
                dump(out/'selective-store.json', uninstall())
                dump(out/'offload-events.json', offload_data)
                uninstall_offload()
        finally:
            if raw is not None:
                raw['gpu_before'] = before
                raw['capture_preemption_mode_label'] = raw.get('preemption_mode')
                raw['preemption_mode'] = config['preemption_mode']
                dump(out/'raw.json', raw)
        dump(out/'memory-after.json', memory_snapshot(engine, torch))
        dump(out/'host-after.json', host_snapshot(engine))
        dump(out/'gpu-after.json', gpu_state())
        metrics = summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5., tpot_slo_s=.2)
        metrics.update(complete_episode_comparison_eligible=raw['status'] == 'COMPLETE',
            actual_preemption_count=raw['actual_preemption_count'],
            forced_rotations=selective['applied_rotations'],
            direct_commits=selective['direct_commits'],
            fit_first_resumes=selective.get('fit_first_resumes', 0),
            capacity_victim_commits=selective.get('capacity_victim_commits', 0),
            role=config['metric_role'])
        dump(out/'metrics.json', metrics)
        reasons = [str(r.get('finish_reason', r.get('stop_reason')) or 'unfinished') for r in raw['requests']]
        dump(out/'status.json', dict(status=raw['status'], capture_status=raw['status'],
            requests_completed=sum(r['status'] == 'completed' for r in raw['requests']),
            finish_reason_counts={reason: reasons.count(reason) for reason in sorted(set(reasons))},
            length_capped_requests=reasons.count('length'),
            forced_rotations=metrics['forced_rotations'], direct_commits=metrics['direct_commits'],
            fit_first_resumes=metrics['fit_first_resumes'],
            capacity_victim_commits=metrics['capacity_victim_commits'],
            scientific_verdict='EXPLORATORY_NATIVE_PAIR', error=raw['error']))
        if raw['status'] != 'COMPLETE':
            raise RuntimeError(raw['error'] or raw['status'])
    except Exception as error:
        dump(out/'status.json', dict(read(out/'status.json'), status='INCOMPLETE', error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        timing['shutdown_start_perf_s']=time.perf_counter()
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        finally:
            timing['process_end_perf_s']=time.perf_counter()
            timing['scope']='Initialization, warmups, capture, later drain and shutdown are distinct host intervals; diagnostic qualification is separate from performance cells.'
            dump(out/'timing.json', timing)


if __name__ == '__main__':
    main()
