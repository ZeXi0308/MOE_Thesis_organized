"""One native store/preempt event. Interface qualification, not a rotation policy."""
import ast
import hashlib
import inspect
import time
from types import MethodType

SCHEDULER_SHA256='2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'


def patch(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
    matches=0
    for i,n in enumerate(fn.body):
        if isinstance(n,ast.Assign) and ast.unparse(n)=='scheduled_timestamp = time.monotonic()':
            fn.body[i+1:i+1]=ast.parse('self._selective_store_begin(preempted_reqs, scheduled_timestamp)').body
            matches+=1
            break
    if matches!=1 or fn.decorator_list:raise ValueError('Unsupported native scheduler shape')
    return ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))


def install(scheduler, *, save, select_step=328):
    connector=scheduler.connector
    if type(connector).__name__!='OffloadingConnector':raise ValueError('Native connector required')
    target=connector.connector_scheduler
    if target.config.offload_prompt_only:raise ValueError('Explicit decode-save config required')
    if scheduler.requests or 'schedule' in vars(scheduler):raise ValueError('Install on drained scheduler')
    manager=scheduler.kv_cache_manager
    if manager.enable_caching or manager.num_kv_cache_groups!=1:
        raise ValueError('Only pinned APC-off one-group qualification')
    original=scheduler.schedule
    source_path=inspect.getsourcefile(type(scheduler))
    with open(source_path,'rb') as f:source=f.read()
    if hashlib.sha256(source).hexdigest()!=SCHEDULER_SHA256:raise ValueError('Native source drift')
    env=dict(original.__func__.__globals__)
    exec(compile(patch(source.decode()),source_path,'exec'),env)
    native=MethodType(env['schedule'],scheduler)
    calc=target._calc_num_offloadable_tokens
    had_calc='_calc_num_offloadable_tokens' in vars(target)
    if '_selective_store_begin' in vars(scheduler):raise ValueError('Hook already installed')
    data=dict(save=save,events=[],selected=None,preempted=False)
    step=0;victim=None;cap=0

    def limited(req_state,num_computed):
        if not save or victim is None or req_state.req is not victim:return 0
        return min(calc(req_state,num_computed),cap)

    def begin(preempted,timestamp):
        nonlocal victim,cap
        if step==select_step:
            if len(scheduler.running)!=31:raise RuntimeError('Unexpected pre-action running set')
            if not all(r.num_output_tokens>0 and r.num_computed_tokens==r.num_tokens-1
                       for r in scheduler.running):raise RuntimeError('Not pure decode state')
            victim=min(scheduler.running,key=lambda r:r.num_output_tokens)
            cap=victim.num_tokens
            data['selected']=victim.request_id
            data['events'].append(dict(event='select',step=step,request=victim.request_id,
                computed=victim.num_computed_tokens,save_cap=cap,output=victim.num_output_tokens))
        if step==select_step+1:
            if victim not in scheduler.running:raise RuntimeError('Victim changed before preemption')
            state=target._req_status.get(victim.request_id)
            jobs=sorted(state.transfer_jobs) if state else []
            if save and not jobs:raise RuntimeError('No registered store job; refuse destructive qualification')
            data['events'].append(dict(event='preempt',step=step,request=victim.request_id,
                pending_jobs=jobs,output=victim.num_output_tokens,time_s=time.perf_counter()))
            scheduler.running.remove(victim)
            scheduler._preempt_request(victim,timestamp)
            preempted.append(victim)
            data['preempted']=True

    def schedule(*args,**kwargs):
        nonlocal step
        result=native(*args,**kwargs)
        if step==select_step and victim.request_id not in result.num_scheduled_tokens:
            raise RuntimeError('Selected request was not scheduled for store creation')
        if step==select_step+1 and victim.request_id not in result.preempted_req_ids:
            raise RuntimeError('Native preemption notification missing')
        if select_step<=step<=select_step+8:
            meta=result.kv_connector_metadata
            data['events'].append(dict(event='metadata',step=step,
                stores=sorted(meta.store_jobs),loads=sorted(meta.load_jobs),flush=sorted(meta.jobs_to_flush)))
        step+=1
        return result

    scheduler._selective_store_begin=begin
    scheduler.schedule=schedule
    target._calc_num_offloadable_tokens=limited

    def uninstall():
        delattr(scheduler,'schedule')
        delattr(scheduler,'_selective_store_begin')
        if had_calc:target._calc_num_offloadable_tokens=calc
        else:delattr(target,'_calc_num_offloadable_tokens')
        data['schedule_calls']=step
        return data
    return data,uninstall
