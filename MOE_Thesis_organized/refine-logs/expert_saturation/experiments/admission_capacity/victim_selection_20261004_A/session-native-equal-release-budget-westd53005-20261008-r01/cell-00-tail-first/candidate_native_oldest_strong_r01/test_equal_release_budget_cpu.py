"""Three targeted CPU checks for the online equal-release budget rule."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'pkg'))
import staged_store_rotation as adapter
from test_partial_restore_once_cpu import native_suffix_action, PINNED


def row(index, output=80, cap=128, computed=3272, release=205):
    return dict(index=index, immediate_releasable_blocks=release, held_blocks=release,
        shared_blocks=0, release_state_error=None, computed_tokens=computed,
        prompt_tokens=computed-output+1, output_tokens=output, max_tokens=cap,
        qualified=True, request_status='RUNNING', pending_native_store_dependencies=0)


def observed():
    return [row(7,100,1024),row(8,100,1024),row(9)]


def real_pick(rule):
    tree=ast.parse((HERE/'pkg/staged_store_rotation.py').read_text())
    install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    pick=deepcopy(next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name=='pick_victim'))
    specs=[(4,1024,19,2),(600,1024,3272,205),(500,1024,3272,205),
           (100,1024,3272,205),(80,128,3272,205)]
    requests=[NS(request_id=str(i),arrival_time=float(i),num_preemptions=0,
        num_output_tokens=output,max_tokens=cap,num_computed_tokens=computed,
        prompt_tokens=computed-output+1,status='RUNNING',spec_token_ids=[7])
        for i,(output,cap,computed,held) in enumerate(specs)]
    owned={};blocks=[];free=[0];freed=[];waiting=[];encoder_freed=[];inflight_discarded=[]
    for req,(_,_,_,held) in zip(requests,specs):
        own=[NS(block_id=len(blocks)+i,ref_cnt=1,is_null=False) for i in range(held)]
        blocks.extend(own);owned[req.request_id]=own
    scheduler=NS(running=requests.copy(),_rotation_full_running_enabled=False,
        waiting=NS(prepend_request=lambda r:waiting.insert(0,r)),reset_preempted_req_ids=set(),
        encoder_cache_manager=NS(free=encoder_freed.append),
        _inflight_prefills=NS(discard=inflight_discarded.append),log_stats=False)
    def release(req):
        assert req not in scheduler.running
        own=owned.pop(req.request_id);free[0]+=len(own);freed.append(req)
        for block in own:block.ref_cnt=0
    scheduler._free_request_blocks=release
    native=deepcopy(next(n for n in ast.walk(ast.parse(PINNED.read_text()))
                        if isinstance(n,ast.FunctionDef) and n.name=='_preempt_request'))
    native.returns=None
    for arg in native.args.args:arg.annotation=None
    native_ns=dict(RequestStatus=NS(RUNNING='RUNNING',PREEMPTED='PREEMPTED'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[native],type_ignores=[])),
                 '<pinned-original-preempt>','exec'),native_ns)
    calls=[]
    def preempt(req,stamp):
        calls.append((req,stamp));native_ns['_preempt_request'](scheduler,req,stamp)
    scheduler._preempt_request=preempt
    once=lambda:dict(mode='SHADOW',consumed=False,trigger_count=0,proposal_count=0,action_request_count=0)
    budget=dict(mode='ACTIVE' if rule=='equal_release_budget' else 'SHADOW',
        decision_count=0,proposal_count=0,action_request_count=0,fallback_counts={},selector_wall_s=0.)
    def view(req):
        return adapter.RequestState(req.request_id,req.num_computed_tokens,req.prompt_tokens,
            req.num_output_tokens,req.max_tokens,req.status,tuple(b.block_id for b in owned[req.request_id]))
    ns=dict(vars(adapter),scheduler=scheduler,victim_rule=rule,full_running_mode='off',
        protected=None,phase=None,current_guard='off',block_size=16,step=12,
        pool=NS(blocks=blocks,get_num_free_blocks=lambda:free[0]),owned=owned,
        residence={r.request_id:(0,0) for r in requests},data=dict(victim_decisions=[]),
        equal_once_state=once(),partial_once_state=once(),host_equal_once_state=once(),
        equal_budget_state=budget,view=view,bidkv_score=lambda r:{},
        cs=NS(_req_status={r.request_id:NS(transfer_jobs=set()) for r in requests},_jobs={}),
        host_prefix=lambda state,*args:dict(host_ready_prefix_blocks=0,
            host_missing_suffix_blocks=state.computed//16,host_state_error=None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pick],type_ignores=[])),
                 '<actual-pick-victim>','exec'),ns)
    scheduler._rotation_pick_victim=ns['pick_victim']
    return scheduler,ns,requests,calls,waiting,freed,encoder_freed,inflight_discarded


class EqualReleaseBudgetTests(unittest.TestCase):
    def test_repeated_max_budget_latest_tie_and_host_independence(self):
        choose=adapter._equal_release_budget_choice
        for _ in range(2):self.assertEqual(choose(observed(),False,16),(8,None,True,2,2))
        rows=observed();rows[0]['max_tokens']=1100
        self.assertEqual(choose(rows,False,16),(7,None,True,2,2))
        for r in rows:r.update(host_state_error='unknown',host_missing_suffix_blocks=None)
        self.assertEqual(choose(rows,False,16),(7,None,True,2,2))
        rows=observed();rows[-1]['max_tokens']=1024;rows[-1]['output_tokens']=100;rows[-1]['prompt_tokens']=3173
        self.assertEqual(choose(rows,False,16),(9,'NO_REMAINING_BUDGET_IMPROVEMENT',False,2,0))

    def test_unknown_protection_capacity_and_controller_guards(self):
        choose=adapter._equal_release_budget_choice
        self.assertEqual(choose(observed(),True,16),(9,'ACTIVE_PROTECTION_OR_PHASE',False,0,0))
        self.assertEqual(choose(observed(),False,None),(9,'UNKNOWN_BLOCK_SIZE',False,0,0))
        for key,reason in [('immediate_releasable_blocks','UNKNOWN_PHYSICAL_RELEASE'),
                ('computed_tokens','UNKNOWN_PROGRESS'),('max_tokens','UNKNOWN_OUTPUT_BUDGET'),
                ('pending_native_store_dependencies','UNKNOWN_PENDING_STORE_STATE')]:
            rows=observed();rows[0][key]=None
            self.assertEqual(choose(rows,False,16),(9,reason,False,0,0))
        for alt in [row(7,100,1024,release=206),row(7,100,1024,computed=3280)]:
            self.assertEqual(choose([alt,observed()[-1]],False,16),
                (9,'NO_EQUAL_RELEASE_AND_COMPUTED_PAGES',False,0,0))
        for changes in [dict(pending_native_store_dependencies=1),dict(qualified=False),
                        dict(shared_blocks=1,held_blocks=206),dict(request_status='PREEMPTED')]:
            rows=observed();rows[-1].update(changes)
            self.assertEqual(choose(rows,False,16),
                (9,'TAIL_NOT_PRIVATE_DECODE_WITHOUT_PENDING_STORE',False,0,0))
        tree=ast.parse((HERE/'pkg/staged_store_rotation.py').read_text())
        install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
        start=next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='victim_rule' for t in n.targets))
        end=next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0])=="data['capacity_deferral_mode']")
        code=compile(ast.Module(body=deepcopy(install.body[start:end]),type_ignores=[]),'<actual-guards>','exec')
        env=dict(A_NATIVE_VICTIM_RULE='equal_release_budget',A_FUNDING_VICTIM_RULE='tail',
            A_NATIVE_VICTIM_FULL_RUNNING='off',A_NATIVE_VICTIM_CURRENT_GUARD='off',
            A_SELF_PREEMPT_CONTINUE='off',A_NATIVE_CAPACITY_DEFERRAL='off')
        for key in [None,'A_NATIVE_VICTIM_FULL_RUNNING','A_NATIVE_VICTIM_CURRENT_GUARD',
                    'A_SELF_PREEMPT_CONTINUE','A_NATIVE_CAPACITY_DEFERRAL']:
            current=dict(env)
            if key:current[key]='prefix_work' if key=='A_NATIVE_CAPACITY_DEFERRAL' else 'on'
            ns=dict(vars(adapter),data={},oldest_admission_mode='queue_fund',prefix_caching=False,
                allow_forced_rotations=True,store_scope='native_full',ordinary_backfill=False)
            with patch.dict(os.environ,current):
                if key:
                    with self.assertRaises(ValueError):exec(code,ns)
                else:exec(code,ns)

    def test_actual_suffix_pop_original_preempt_lifecycle_and_repeated_action(self):
        execute=native_suffix_action()
        for rule,indices in [('equal_release_budget',[3,2]),('tail',[4,3])]:
            scheduler,ns,requests,calls,waiting,freed,enc,discarded=real_pick(rule)
            for n,index in enumerate(indices):
                stamp=object();selected=execute(scheduler,1,[requests[0]],stamp)
                self.assertIs(selected,requests[index]);self.assertNotIn(selected,scheduler.running)
                self.assertEqual(calls[-1],(selected,stamp));self.assertEqual(len(calls),n+1)
                self.assertEqual((selected.status,selected.num_computed_tokens,selected.spec_token_ids,
                                  selected.num_preemptions),('PREEMPTED',0,[],1))
                self.assertIs(waiting[0],selected);self.assertIs(freed[-1],selected)
                self.assertIs(enc[-1],selected);self.assertIs(discarded[-1],selected)
                self.assertIn(selected.request_id,scheduler.reset_preempted_req_ids)
                self.assertNotIn(selected.request_id,ns['residence'])
                decision=ns['data']['victim_decisions'][-1];proposal=decision['equal_release_budget']
                self.assertEqual([r['index'] for r in decision['candidates']],list(range(1,5-n)))
                self.assertEqual(proposal['returned_request'],selected.request_id)
                self.assertEqual(proposal['action_requested'],rule=='equal_release_budget')
                self.assertEqual(proposal['tail_releasable_blocks'],205)
                self.assertEqual(proposal['proposed_releasable_blocks'],205)
                self.assertGreaterEqual(proposal['selector_wall_s'],0.)
                for shadow in ['equal_held_once','partial_restore_once','equal_release_host_once','max_release_shadow']:
                    self.assertIsNotNone(decision[shadow])
            self.assertEqual(ns['equal_budget_state']['decision_count'],2)
            self.assertEqual(ns['equal_budget_state']['action_request_count'],2 if rule=='equal_release_budget' else 0)
            self.assertEqual(ns['equal_budget_state']['proposal_count'],2 if rule=='equal_release_budget' else 1)
        for name in ['rotation_native.py','request_measurement.py']:
            self.assertEqual((HERE/'pkg'/name).read_bytes(),
                (HERE.parent/'candidate_native_equal_release_host_once_r01/pkg'/name).read_bytes())


if __name__=='__main__':unittest.main()
