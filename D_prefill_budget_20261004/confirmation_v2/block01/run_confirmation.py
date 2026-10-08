"""Serial real vLLM continuous serving comparison, one shared compilation domain."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
UUID = os.environ.get('D_GPU_UUID', 'GPU-bf3fc5ab-804d-b9d6-759b-4390899f15b9')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def episode(engine, work, policy_name, path, target_ms):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from prefill_policy import Policy
    path.mkdir()
    scheduler = engine.engine_core.engine_core.scheduler
    assert not scheduler.requests and not engine.has_unfinished_requests()
    policy = Policy(policy_name, target_ms)
    native = scheduler.schedule
    rows = {x['request_id']: dict(x, prompt_tokens=len(x['prompt_token_ids']),
        add_s=None, first_scheduled_s=None, completion_s=None, finished=False,
        token_times_s=[], output_token_ids=[]) for x in work}
    steps = []
    origin = time.perf_counter()
    epoch = time.time()
    def now(): return time.perf_counter()-origin
    def schedule(*args, **kwargs):
        started = now()
        decoding = [r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens]
        decode_ids = {r.request_id for r in decoding}
        contexts = sum(r.num_computed_tokens for r in decoding)
        t = time.perf_counter()
        budget = policy.choose(len(decoding))
        scheduler.d_prefill_budget = budget
        overhead = (time.perf_counter()-t)*1e6
        before = {rid:(r.num_computed_tokens,r.num_prompt_tokens,
                  engine.output_processor.request_states[rid].external_req_id)
                  for rid,r in scheduler.requests.items()}
        out = native(*args, **kwargs)
        detail = []
        for rid, amount in out.num_scheduled_tokens.items():
            previous, prompt, external = before[rid]
            start = scheduler.requests[rid].num_computed_tokens-amount
            prefill = min(amount,max(0,prompt-start))
            row = rows[external]
            if row['first_scheduled_s'] is None: row['first_scheduled_s'] = started
            detail.append(dict(request_id=external, prefill_tokens=prefill,
                               decode_tokens=amount-prefill, computed_start=start))
        pt = sum(r['prefill_tokens'] for r in detail)
        dt = sum(r['decode_tokens'] for r in detail)
        assert pt <= budget
        assert not out.preempted_req_ids, 'First experiment requires zero KV pressure/preemption'
        assert decode_ids <= set(out.num_scheduled_tokens), 'Native decoder unexpectedly skipped'
        assert all(out.num_scheduled_tokens[rid] == 1 for rid in decode_ids)
        steps.append(dict(start_s=started, end_s=None, budget=budget, prefill_tokens=pt,
            prefill_backlog_tokens_before=sum(max(0,prompt-previous) for previous,prompt,external in before.values()),
            prefill_backlog_requests_before=sum(previous<prompt for previous,prompt,external in before.values()),
            decode_tokens=dt, decode_count_before=len(decoding), decode_context_sum=contexts,
            schedule_s=now()-started, decision_us=overhead, requests=detail,
            running_count=len(scheduler.running),waiting_count=len(scheduler.waiting),
            kv_used_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1-scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            kv_total_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1,
            preempted=list(out.preempted_req_ids or [])))
        return out
    scheduler.schedule = schedule
    pending = sorted(work, key=lambda x:x['arrival_s'])
    pos = 0
    try:
        while pos < len(pending) or engine.has_unfinished_requests():
            if now()>180: raise TimeoutError('Complete-service bound exceeded; retain partial results')
            while pos<len(pending) and pending[pos]['arrival_s']<=now():
                item = pending[pos]
                p = SamplingParams(temperature=0, max_tokens=item['max_tokens'],
                    min_tokens=item['max_tokens'], ignore_eos=True, detokenize=False,
                    output_kind=RequestOutputKind.CUMULATIVE)
                engine.add_request(item['request_id'], {'prompt_token_ids':item['prompt_token_ids']},
                                   p, arrival_time=epoch+item['arrival_s'])
                rows[item['request_id']]['add_s']=now()
                pos += 1
            if not engine.has_unfinished_requests():
                time.sleep(min(.002,max(0,pending[pos]['arrival_s']-now())))
                continue
            start = now()
            before_count = len(steps)
            outputs = engine.step()
            end = now()
            assert len(steps)==before_count+1
            step = steps[-1]
            step.update(start_s=start,end_s=end)
            policy.observe(end-start,step['prefill_tokens'],step['decode_tokens'])
            for output in outputs:
                row = rows[output.request_id]
                tokens = list(output.outputs[0].token_ids)
                old = len(row['output_token_ids'])
                assert tokens[:old]==row['output_token_ids']
                row['token_times_s'].extend([end]*(len(tokens)-old))
                row['output_token_ids']=tokens
                if output.finished:
                    row.update(finished=True,completion_s=end,finish_reason=output.outputs[0].finish_reason)
        assert all(r['finished'] and len(r['output_token_ids'])==r['max_tokens'] for r in rows.values())
        assert not scheduler.requests
    finally:
        scheduler.schedule = native
        data = dict(policy=policy_name,elapsed_s=now(),requests=list(rows.values()),steps=steps)
        dump(path/'raw.json',data)
    return dict(policy=policy_name,elapsed_s=data['elapsed_s'],requests=len(rows),
                output_tokens=sum(len(r['output_token_ids']) for r in rows.values()),
                mixed_steps=sum(s['prefill_tokens']>0 and s['decode_tokens']>0 for s in steps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workload',default='workload_dev.json')
    p.add_argument('--policies',default='fixed512,fixed2048,fixed2048,fixed512')
    p.add_argument('--target-ms',type=float,default=60)
    p.add_argument('--warm-policies',default='fixed304,fixed384,decode_v2,feedback')
    p.add_argument('--wait-lock',type=float,default=0)
    p.add_argument('--command-loop',action='store_true')
    a = p.parse_args()
    lock = open('/root/autodl-tmp/moe-research-gpu.lock','a+')
    if a.wait_lock > 0:
        def lock_timeout(signum, frame):
            raise TimeoutError('Shared GPU lock deadline exceeded')
        previous_alarm = signal.signal(signal.SIGALRM, lock_timeout)
        signal.setitimer(signal.ITIMER_REAL, a.wait_lock)
        print('WAITING_SHARED_LOCK_NO_GPU_INITIALIZED',flush=True)
        try:
            fcntl.flock(lock,fcntl.LOCK_EX)
        except TimeoutError:
            print('LOCK_BUSY_NO_GPU_INITIALIZED',flush=True)
            return 75
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
    else:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('LOCK_BUSY_NO_GPU_INITIALIZED',flush=True)
            return 75
    state = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    if state: raise RuntimeError('GPU occupied: '+state)
    a.output.mkdir(exist_ok=False)
    out = a.output
    dump(out/'status.json',dict(status='INITIALIZING',pid=os.getpid()))
    os.environ.update(CUDA_VISIBLE_DEVICES=UUID,VLLM_USE_FLASHINFER_SAMPLER='0',
        OMP_NUM_THREADS='8',VLLM_ENABLE_V1_MULTIPROCESSING='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        from runtime_bootstrap import apply
        repair = apply()
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        from prefill_policy import install
        torch.set_num_threads(8)
        work = json.loads((ROOT/a.workload).read_text())
        kwargs = dict(model='/root/autodl-tmp/moe-research-20261002/model',dtype='bfloat16',
            seed=20261004,max_model_len=4096,max_num_seqs=192,max_num_batched_tokens=4096,
            long_prefill_token_threshold=0,enable_chunked_prefill=True,enable_prefix_caching=False,
            scheduling_policy='fcfs',async_scheduling=False,
            scheduler_reserve_full_isl=True,stream_interval=1,enforce_eager=False,
            enable_return_routed_experts=False,gpu_memory_utilization=.9)
        dump(out/'engine_args.json',kwargs)
        protocol = dict(policies=a.policies.split(','),request_count=len(work),workload_sha256=hashlib.sha256((ROOT/a.workload).read_bytes()).hexdigest(),
            slo=dict(ttft_s=4,gap_s=.1,completion_s=20), target_ms=a.target_ms,
            output_contract='Fixed lengths and ignore_eos for scheduling causality; not natural ending quality evaluation.',
            warmup='Each policy replays the same confirmation trace before measurement; same process, drained engine, no APC',
            warm_policies=a.warm_policies.split(','),
            timing='Host engine.step completion, no extra GPU synchronize; includes scheduler and output handling',
            source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
        dump(out/'protocol.json',protocol)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs),enable_multiprocessing=False)
        scheduler=engine.engine_core.engine_core.scheduler
        patch=install(scheduler)
        (out/'patched_schedule.py').write_text(patch.pop('source'))
        dump(out/'patch.json',patch)
        dump(out/'environment.json',dict(torch=torch.__version__,vllm=vllm.__version__,cuda=torch.version.cuda,repair=repair,
             kv_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks,
             kv_block_size=engine.vllm_config.cache_config.block_size,
             gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],text=True)))
        for policy_name in a.warm_policies.split(','):
            print('WARMUP',policy_name,flush=True)
            episode(engine,work,policy_name,out/('warm_'+policy_name),a.target_ms)
        summaries=[]
        for i,policy in enumerate(a.policies.split(',')):
            print('MEASURE',i,policy,flush=True)
            result=episode(engine,work,policy,out/f'{i:02d}_{policy}',a.target_ms)
            summaries.append(result)
            print(json.dumps(result),flush=True)
        dump(out/'status.json',dict(status='COMPLETE',runs=summaries))
        command_index=0
        while a.command_loop:
            command_path=ROOT/f'normal-command-{command_index:02d}.json'
            print('AWAIT_COMMAND',str(command_path),flush=True)
            deadline=time.monotonic()+90
            while not command_path.exists() and time.monotonic()<deadline:
                time.sleep(1)
            if not command_path.exists(): break
            command=json.loads(command_path.read_text())
            if command.get('stop'): break
            phase=ROOT/command['output']
            phase.mkdir(exist_ok=False)
            phase_work=json.loads((ROOT/command.get('workload',a.workload)).read_text())
            phase_protocol=dict(protocol,policies=command['policies'],request_count=len(phase_work),
                workload_sha256=hashlib.sha256((ROOT/command.get('workload',a.workload)).read_bytes()).hexdigest(),
                target_ms=command.get('target_ms',a.target_ms),command=command,
                source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
            dump(phase/'protocol.json',phase_protocol)
            phase_summaries=[]
            if command.get('warmup',False):
                for budget in (256,512,1024,2048):
                    episode(engine,phase_work,'fixed'+str(budget),phase/('warm'+str(budget)),phase_protocol['target_ms'])
            for i,policy in enumerate(command['policies']):
                print('MEASURE_PHASE',command_index,i,policy,flush=True)
                result=episode(engine,phase_work,policy,phase/f'{i:02d}_{policy}',phase_protocol['target_ms'])
                phase_summaries.append(result)
                print(json.dumps(result),flush=True)
            dump(phase/'status.json',dict(status='COMPLETE',runs=phase_summaries))
            command_index+=1
    except Exception:
        dump(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
