"""Synthetic boundary checks; fixtures are not native GPU performance evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import analyze as base
import analyze_first_fork as audit


def write(path, data):
    path.write_text(json.dumps(data))


def fixture(group):
    parent=group/'00_same_engine';parent.mkdir()
    for p in (group,parent): write(p/'status.json',dict(status='COMPLETE'))
    prompt=list(range(16)); digest=hashlib.sha256(json.dumps(prompt+[4]).encode()).hexdigest()
    cells=[]
    for index,policy in enumerate(('host','recompute')):
        cell=parent/f'{index:02d}_{policy}';cell.mkdir();cells.append(cell)
        rid=f'random-{index}';event=100+600*index
        decision=dict(event=event, request_id=rid, preemptions=1, decision_s=2.85,
            known_tokens=17,prompt_tokens=16,generated_tokens=1,prefix_sha256=digest,
            host_hit_tokens=16, free_blocks=10,full_required_blocks=2,reserved_blocks=0,watermark_blocks=0,
            joint_capacity=True,eligible=True,request_state_preserved=True,pending_load_jobs=0,
            pending_transfer_jobs=0,running=0,waiting=1,action=policy,policy=policy,fallback='none',recent_step_s=.2)
        commit=dict(decision,allocation_s=2.9,actual_action=policy,external_tokens=16 if index==0 else 0,
                    allocated_blocks=1 if index==0 else 2)
        steps=[dict(start_s=1.8,end_s=2.,process_cpu_s=.2,driver_thread_cpu_s=.2),
               dict(start_s=2.8,end_s=3.,process_cpu_s=.2,driver_thread_cpu_s=.2)]
        entry=lambda count,start,end,known,gen:dict(request_id=rid,count=count,start_computed=start,
            end_computed=end,known_tokens=known,generated_tokens=gen)
        schedule=[dict(time_s=1.9,free_blocks_after_schedule=99,scheduled=[entry(16,0,16,16,0)]),
                  dict(time_s=2.95,free_blocks_after_schedule=8,scheduled=[] if index==0 else [entry(17,0,17,17,1)])]
        completion=4. if index==0 else 3.
        if index==0:
            steps.append(dict(start_s=3.8,end_s=4.,process_cpu_s=.2,driver_thread_cpu_s=.2))
            schedule.append(dict(time_s=3.9,free_blocks_after_schedule=8,scheduled=[entry(1,16,17,17,1)]))
        raw=dict(requests=[dict(request_id=rid,external_id='measured/E000',arrival_s=0.,prompt_tokens=16,
            max_output_tokens=2,output_token_ids=[4,5],token_times_s=[2.,completion],completed=True,
            completion_s=completion,finish_reason='stop',stop_reason=None)], steps=steps,scheduler_steps=schedule,
            decisions=[decision],commits=[commit],preemptions=[dict(request_id=rid,time_s=2.5,known_tokens=17,
                generated_tokens=1,computed_tokens=16,free_blocks_before_preempt=0)],transfers=[],
            all_complete_s=completion+.001,service_and_drain_s=completion+.002,
            target_selection=dict(selection_mode='full_policy'))
        files={'raw.json':raw,'status.json':dict(status='COMPLETE'),
            'config.json':dict(out=str(cell),policy=policy,policies='host,recompute',threshold=1792,target_spec=None),
            'inputs.json':[dict(prompt_token_ids=prompt,output_tokens=2,arrival_s=0.)],
            'engine_args.json':dict(max_num_seqs=320),
            'resources.json':dict(gpu_blocks=100,group_config=[dict(tokens_per_block=16)]),
            'warmup.json':dict(requests=[])}
        for name,value in files.items():write(cell/name,value)
    summary=group/'summary.json';write(summary,base.analyze_group(group))
    return cells,summary


def update_raw(cell, change):
    path=cell/'raw.json';raw=json.loads(path.read_text());change(raw);write(path,raw)


class FirstForkTests(unittest.TestCase):
    def test_first_joint_commit_matches_history_not_event_number(self):
        with tempfile.TemporaryDirectory() as folder:
            cells,summary=fixture(Path(folder));r=audit.analyze(*cells,summary)
            self.assertEqual(r['diagnostic_status'],'MATCHED_FIRST_JOINT_FORK',r)
            self.assertTrue(r['joint_fork_established'])
            self.assertEqual([x['event'] for x in r['candidate']['commit_boundaries']],[100,700])
            self.assertEqual(r['candidate']['identity']['external_id'],'measured/E000')
            self.assertEqual(r['normalized_native_trace']['first_difference']['index'],1)
            self.assertEqual(r['step_output_emissions']['first_difference']['index'],1)

    def test_earlier_uncommitted_lookup_difference_is_not_erased(self):
        with tempfile.TemporaryDirectory() as folder:
            cells,summary=fixture(Path(folder))
            for c in cells:
                def insert(raw):
                    earlier=dict(raw['decisions'][0],decision_s=2.82,event=raw['decisions'][0]['event']-1)
                    raw['decisions'].insert(0,earlier)
                update_raw(c,insert)
            write(summary,base.analyze_group(Path(folder)))
            r=audit.analyze(*cells,summary)
            self.assertEqual(r['diagnostic_status'],'PRIOR_LOOKUP_DIVERGENCE')
            self.assertFalse(r['joint_fork_established'])
            self.assertTrue(r['candidate']['prior_uncommitted_different_lookup'])
            self.assertFalse(r['first_lookup_boundaries'][0]['committed'])
            self.assertEqual(r['candidate']['commit_boundaries'][0]['lookup_index'],1)

    def test_zero_domain_is_explicit_and_not_inferred_from_step_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            cells,summary=fixture(Path(folder))
            for c in cells:update_raw(c,lambda r:r.update(decisions=[],commits=[]))
            write(summary,base.analyze_group(Path(folder)))
            r=audit.analyze(*cells,summary)
            self.assertEqual(r['diagnostic_status'],'NO_JOINT_ELIGIBLE_COMMITS')
            self.assertFalse(r['joint_fork_established'])
            self.assertFalse(r['normalized_native_trace']['equal'])

    def test_same_event_number_does_not_match_different_external_request(self):
        with tempfile.TemporaryDirectory() as folder:
            cells,summary=fixture(Path(folder))
            def change(raw):
                raw['requests'][0]['external_id']='measured/E009'
                for e in raw['decisions']+raw['commits']:e['event']=100
            update_raw(cells[1],change);write(summary,base.analyze_group(Path(folder)))
            r=audit.analyze(*cells,summary)
            self.assertEqual(r['diagnostic_status'],'UNMATCHED_FIRST_COMMIT_EXECUTION')
            self.assertIsNone(r['candidate'])

    def test_earlier_state_difference_does_not_hide_later_uncommitted_advice(self):
        with tempfile.TemporaryDirectory() as folder:
            cells,summary=fixture(Path(folder))
            for index,c in enumerate(cells):
                def insert(raw):
                    d=raw['decisions'][0]
                    state_only=dict(d,event=d['event']-2,decision_s=2.81,eligible=False,joint_capacity=False,
                                    action='native',fallback='pending_native',host_hit_tokens=None,free_blocks=10+index)
                    advice=dict(d,event=d['event']-1,decision_s=2.82)
                    raw['decisions'][:0]=[state_only,advice]
                update_raw(c,insert)
            write(summary,base.analyze_group(Path(folder)))
            r=audit.analyze(*cells,summary)
            self.assertEqual(r['first_lookup_boundaries'][0]['action'],'native')
            self.assertTrue(r['candidate']['prior_uncommitted_different_lookup'])
            self.assertEqual(len(r['candidate']['earlier_uncommitted_advice_differences']),2)
            self.assertTrue(all(x['boundary']['lookup_index']==1 for x in r['candidate']['earlier_uncommitted_advice_differences']))

    def test_hash_capacity_and_commit_link_failures_are_not_legal_forks(self):
        for case in ('hash','capacity','actual','orphan','failed_parent'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as folder:
                cells,summary=fixture(Path(folder))
                if case=='failed_parent':write(cells[0].parent/'status.json',dict(status='FAILED'))
                else:
                    def change(raw):
                        if case=='hash':
                            for e in raw['decisions']+raw['commits']:e['prefix_sha256']='0'*64
                        elif case=='capacity':
                            for e in raw['decisions']+raw['commits']:e['free_blocks']=1
                        elif case=='actual':raw['commits'][0]['actual_action']='recompute'
                        else:raw['commits'][0]['event']+=3
                    update_raw(cells[0],change)
                with self.assertRaises(ValueError):audit.analyze(*cells,summary)

    def test_visible_state_or_prior_schedule_difference_remains_unmatched(self):
        for case in ('state','schedule'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as folder:
                cells,summary=fixture(Path(folder))
                def change(raw):
                    if case=='state':
                        for e in raw['decisions']+raw['commits']:e['free_blocks']=11
                    else:raw['scheduler_steps'][0]['free_blocks_after_schedule']=98
                update_raw(cells[1],change)
                r=audit.analyze(*cells,summary)
                self.assertEqual(r['diagnostic_status'],'JOINT_FORK_WITH_STATE_OR_PREHISTORY_MISMATCH')
                self.assertFalse(r['joint_fork_established'])


if __name__=='__main__':unittest.main()
