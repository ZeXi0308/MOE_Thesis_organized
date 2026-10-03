"""Run sealed native queue selection/promotion and patched loop prefix on CPU.

Stops before allocation. No allocator, connector or worker execution is claimed.
"""
import ast
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from rotation_native import patched_schedule_tree, SCHEDULER_SHA256

class Status(Enum):
    PREEMPTED=1
    WAITING_FOR_REMOTE_KVS=2
    WAITING=3
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR=4
    WAITING_FOR_STREAMING_REQ=5

class Queue(list):
    def peek_request(self):return self[0]
    def pop_request(self):return self.pop(0)
    def prepend_request(self,r):self.insert(0,r)


def check():
    path=Path('refine-logs/expert_saturation/outputs/admission_capacity/20260914_load_ready_contract_r01/native_source.json')
    source=json.loads(path.read_text())['v1/core/sched/scheduler.py']
    assert hashlib.sha256(source.encode()).hexdigest()==SCHEDULER_SHA256
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    names={'_select_waiting_queue_for_scheduling','_try_promote_blocked_waiting_request'}
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    patched=patched_schedule_tree(source)
    loop=next(n for n in ast.walk(patched) if isinstance(n,ast.While) and ast.unparse(n.test)=='(self.waiting or self.skipped_waiting) and token_budget > 0')
    # Preserve the native and injected prefix through blocked-request handling.
    end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and '_try_promote_blocked_waiting_request' in ast.unparse(n))
    loop.body=loop.body[:end+1]+ast.parse('eligible.append(request.request_id)\nbreak').body
    probe=ast.parse('def probe(self):\n token_budget=1024\n step_skipped_waiting=Queue()\n eligible=[]\n').body[0]
    probe.body.append(loop)
    probe.body+=ast.parse('return eligible, [r.request_id for r in step_skipped_waiting]').body
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='Native',bases=[],keywords=[],body=methods+[probe],decorator_list=[])],type_ignores=[])
    env=dict(RequestStatus=Status,SchedulingPolicy=NS(FCFS='FCFS'),Queue=Queue,logger=NS(debug=lambda *a:None))
    exec(compile(ast.fix_missing_locations(module),'<native-waiting-prefix>','exec'),env)
    rows=[]
    for ready in (False,True):
        obj=env['Native']();obj.policy='FCFS';obj.running=[];obj.max_num_running_reqs=32
        obj.num_waiting_for_streaming_input=0;obj._rotation_target='target'
        req=NS(request_id='target',status=Status.WAITING_FOR_REMOTE_KVS,num_preemptions=1)
        obj.skipped_waiting=Queue([req]);obj.waiting=Queue([NS(request_id='victim',status=Status.PREEMPTED)])
        obj.finished_recving_kv_req_ids={'target'} if ready else set()
        promoted=[]
        obj._update_waiting_for_remote_kv=lambda r:promoted.append(r.request_id)
        obj._is_blocked_waiting_status=lambda s:s==Status.WAITING_FOR_REMOTE_KVS
        eligible,skipped=obj.probe()
        assert eligible==(['target'] if ready else [])
        assert skipped==([] if ready else ['target'])
        assert promoted==(['target'] if ready else [])
        assert obj.waiting[0].request_id=='victim'
        rows.append(dict(completion_visible=ready,eligible=eligible,skipped=skipped,promoted=promoted))
    return dict(status='CPU_NATIVE_WAITING_PREFIX_PASS',cases=rows,
        scope='Actual native selection and promotion methods plus patched waiting-loop prefix; fake queues and cache-completion callback. Allocation, complete schedule, worker and GPU remain untested.')

if __name__=='__main__':print(json.dumps(check(),indent=2))
