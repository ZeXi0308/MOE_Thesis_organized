"""CPU protocol checks with a deterministic scheduler/clock, not GPU timings.

The fake preserves a running-request cap, FCFS admission after completions and
different request completion times. It does not model GPU overlap, KV or vLLM
sampling, so these tests establish runner semantics only.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matrix_under_test', ROOT/'code/run_matrix.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProtocolTests(unittest.TestCase):
    def test_cap_contract_and_legacy_default(self):
        self.assertEqual(runner.resolve_admission({'batches':[1,16,64]},[1,16,64]),('wave',64))
        self.assertEqual(runner.resolve_admission({'batches':[16],'admission_mode':'queued'},[16]),('queued',16))
        for plan,batches in [
            ({'batches':[16,64],'admission_mode':'queued'},[16]),
            ({'batches':[16],'admission_mode':'queued','max_num_seqs':64},[16]),
            ({'batches':[16],'admission_mode':'queued'},[8]),
            ({'batches':[16],'max_num_seqs':8},[16]),
            ({'batches':[16],'measured_requests':0},[16]),
        ]:
            with self.subTest(plan=plan,batches=batches),self.assertRaises(ValueError):
                runner.resolve_admission(plan,batches)

    def simulate(self, admission_mode=None, shape_warmup_offset=None):
        clock=NS(ns=0)
        def now():
            clock.ns+=1_000_000
            return clock.ns
        runtime=NS(label=None,steps=[],events=[],acceptance=NS(snapshot=lambda:{'alpha':0.7}))
        def start(label,trace=False):
            runtime.label=label
            runtime.steps=[]
        def end():
            clock.ns+=11_000_000  # Deliberately visible inter-wave CPU summary.
            return {'steps':runtime.steps,'routes':[],
                    'summary':{'trace_complete':True,'route_errors':[]}}
        runtime.start_episode=start
        runtime.end_episode=end
        runtime.attach_engine=lambda engine:None

        class Engine:
            def __init__(self,args):
                self.args=args
                self.waiting=[]
                self.running=[]
                self.engine_core=NS(shutdown=lambda:None)
            def add_request(self,rid,prompt,params):
                self.waiting.append(NS(rid=rid,source=prompt['prompt_token_ids'][0],
                                       maximum=params.max_tokens,ids=[]))
                runtime.events.append((runtime.label,'submit',rid))
                clock.ns+=2_000_000
            def has_unfinished_requests(self):
                return bool(self.waiting or self.running)
            def step(self):
                while self.waiting and len(self.running)<self.args.max_num_seqs:
                    self.running.append(self.waiting.pop(0))
                active=list(self.running)
                runtime.events.append((runtime.label,'step',[r.rid for r in active]))
                runtime.steps.append({'step':len(runtime.steps),'episode':runtime.label,
                    'phase':'decode','decode_batch':len(active),
                    'scheduled_tokens_by_request':{r.rid:1 for r in active}})
                output=[]
                for request in active:
                    # Even source IDs advance faster, making backfill observable.
                    count=min(2 if request.source%2==0 else 1,request.maximum-len(request.ids))
                    request.ids.extend([7]*count)
                    finished=len(request.ids)==request.maximum
                    output.append(NS(request_id=request.rid,finished=finished,
                        outputs=[NS(token_ids=list(request.ids),finish_reason='length' if finished else None,stop_reason=None)]))
                    if finished:
                        self.running.remove(request)
                clock.ns+=10_000_000
                return output

        engines=[]
        def create(args,**kwargs):
            self.assertEqual(os.environ['VLLM_USE_V2_MODEL_RUNNER'],'0')
            engine=Engine(args)
            engines.append(engine)
            return engine
        class Meter:
            def __init__(self,interval_s):pass
            def start(self):clock.ns+=3_000_000
            def stop(self):
                clock.ns+=7_000_000
                return {'energy_j':1.0}
            def close(self):pass
        def module(name,**attributes):
            item=ModuleType(name)
            item.__dict__.update(attributes)
            return item
        modules={
            'torch':module('torch',__version__='fake',version=NS(cuda='fake'),set_num_threads=lambda _:None,
                cuda=NS(synchronize=lambda:None,get_device_name=lambda _:'fake CPU protocol',memory_allocated=lambda:0)),
            'vllm':module('vllm',__version__='0.26.0',SamplingParams=lambda **kw:NS(**kw)),
            'vllm.engine.arg_utils':module('vllm.engine.arg_utils',EngineArgs=lambda **kw:NS(**kw)),
            'vllm.v1.engine.llm_engine':module('vllm.v1.engine.llm_engine',LLMEngine=NS(from_engine_args=create)),
            'vllm.sampling_params':module('vllm.sampling_params',RequestOutputKind=NS(CUMULATIVE='cumulative')),
            'vllm_s_plugin':module('vllm_s_plugin',install=lambda config:runtime),
            'cost_model':module('cost_model',AcceptanceEWMA=lambda:NS()),
            'power':module('power',PowerSampler=Meter),
        }
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            dataset=base/'workload.json'
            dataset.write_text(json.dumps({'records':[{'source_index':i,'prompt_token_ids':[i]} for i in range(8)]}))
            plan={'model':'fake','workload':str(dataset),'draft_source':'ngram','batches':[2],
                'repeats':2,'measured_requests':4,'output_tokens':4,'warmup_tokens':4,
                'warmup_offset':4,'warmup_each_batch':True,'warmup_full_population':True,
                'trace_tokens':2,'skip_trace':True,'max_num_seqs':2}
            if admission_mode is not None:
                plan['admission_mode']=admission_mode
            if shape_warmup_offset is not None:
                plan['shape_warmup_offset']=shape_warmup_offset
            config=base/'plan.json'
            config.write_text(json.dumps(plan))
            output=base/'output'
            with patch.dict(sys.modules,modules),patch.dict(os.environ,{'MOE_S_GPU_LOCKED':'1'}), \
                 patch.object(sys,'argv',['run_matrix','--config',str(config),'--gamma','0','--output',str(output)]), \
                 patch.object(runner.time,'perf_counter_ns',now),contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            results={path.name:json.loads(path.read_text()) for path in output.glob('*.json')}
        return results,runtime.events,engines[0].args

    def assert_times(self,result):
        for episode in result['episodes']:
            offset=episode['population_window_start_elapsed_s']
            for request in episode['requests']:
                self.assertEqual(request['release_elapsed_s'],0)
                self.assertLessEqual(request['submit_elapsed_s'],request['submit_complete_elapsed_s'])
                self.assertLessEqual(request['submit_complete_elapsed_s'],request['first_output_elapsed_s'])
                self.assertLessEqual(request['first_output_elapsed_s'],request['last_output_elapsed_s'])
                self.assertLessEqual(request['last_output_elapsed_s'],request['completion_elapsed_s'])
                self.assertAlmostEqual(request['first_output_elapsed_s'],offset+request['chunks'][0]['elapsed_s'])
                self.assertAlmostEqual(request['last_output_elapsed_s'],offset+request['chunks'][-1]['elapsed_s'])
                self.assertEqual(len(request['output_token_ids']),4)
            for host in episode['host_steps']:
                self.assertAlmostEqual(host['population_elapsed_s'],offset+host['elapsed_s'])

    def test_queued_submits_once_and_backfills_under_cap(self):
        outputs,events,args=self.simulate('queued')
        self.assertEqual(args.max_num_seqs,2)
        for repeat in (0,1):
            result=outputs[f'perf-r{repeat}-b2.json']
            self.assertEqual(result['admission_mode'],'queued')
            self.assertEqual(result['batch'],2)
            self.assertEqual(result['population'],4)
            self.assertEqual(len(result['episodes']),1)
            self.assertEqual(result['wall_s'],result['population_wall_s'])
            self.assertEqual(result['population_inter_episode_gap_s'],0)
            self.assert_times(result)
            flow=[event for event in events if event[0]==f'perf-r{repeat}-b2-q0']
            self.assertEqual([event[1] for event in flow[:4]],['submit']*4)
            active=[event[2] for event in flow if event[1]=='step']
            self.assertTrue(all(len(ids)<=2 for ids in active))
            self.assertTrue(any(any(r.endswith('-1') for r in ids) and
                                any(r.endswith('-2') for r in ids) for ids in active))
        self.assertFalse(any(name.startswith('routes-') for name in outputs))

    def test_legacy_wave_preserves_windows_and_records_local_wait(self):
        outputs,events,_=self.simulate()
        for repeat in (0,1):
            result=outputs[f'perf-r{repeat}-b2.json']
            self.assertEqual(result['admission_mode'],'wave')
            self.assertEqual(len(result['episodes']),2)
            self.assertEqual(result['wall_s'],sum(e['wall_s'] for e in result['episodes']))
            self.assertEqual(result['energy_j'],2)
            self.assertGreater(result['population_inter_episode_gap_s'],0)
            self.assertAlmostEqual(result['population_wall_s'],result['wall_s']+result['population_inter_episode_gap_s'])
            self.assert_times(result)
            first,second=result['episodes']
            self.assertGreater(min(r['submit_elapsed_s'] for r in second['requests']),
                               max(r['completion_elapsed_s'] for r in first['requests']))
            self.assertEqual(result['tokens_per_second'],16/result['wall_s'])
            self.assertEqual(result['population_tokens_per_second'],16/result['population_wall_s'])

    def test_matched_diagnostic_configs_and_warmup(self):
        queued=json.loads((ROOT/'configs/olmoe_backfill_diagnostic.json').read_text())
        wave=json.loads((ROOT/'configs/olmoe_wave_cap16_diagnostic.json').read_text())
        for key in queued:
            if key not in ('admission_mode','measurement'):
                self.assertEqual(queued[key],wave[key],key)
        self.assertEqual(runner.resolve_admission(queued,queued['batches']),('queued',16))
        self.assertEqual(runner.resolve_admission(wave,wave['batches']),('wave',16))
        queued_outputs,queued_events,_=self.simulate('queued')
        wave_outputs,wave_events,_=self.simulate('wave')
        def warm(events):return [event for event in events if not event[0].startswith('perf-')]
        self.assertEqual(warm(queued_events),warm(wave_events))
        self.assertEqual(queued_outputs['population-shape-warmup-b2.json']['admission_mode'],'queued')
        self.assertEqual(wave_outputs['population-shape-warmup-b2.json']['admission_mode'],'queued')

    def test_disjoint_shape_warmup_population(self):
        outputs,_,_=self.simulate(shape_warmup_offset=4)
        warm_ids={row['source_index'] for name,result in outputs.items()
                  if name.startswith(('shape-warmup-','population-shape-warmup-'))
                  for row in result['requests']}
        measured_ids={row['source_index'] for row in outputs['perf-r0-b2.json']['requests']}
        self.assertEqual(warm_ids,{4,5,6,7})
        self.assertEqual(measured_ids,{0,1,2,3})
        self.assertEqual(outputs['evaluation_initial_state.json']['shape_warmup_offset'],4)
        self.assertEqual(outputs['evaluation_initial_state.json']['acceptance_snapshot_before'],{'alpha':0.7})


if __name__=='__main__':
    unittest.main()
