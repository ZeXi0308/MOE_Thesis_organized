"""Bounded healthy-input adapter checks; no vLLM installation or GPU needed."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PKG = ROOT/'pkg'


def module(name):
    spec = importlib.util.spec_from_file_location(name, PKG/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RUNNER = module('run_recovery_cadence')
SAFE = module('safe_static')
MEASURE = module('request_measurement')
TREE = ast.parse((PKG/'run_recovery_cadence.py').read_text())
MAIN = next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')
BODY = next(n for n in MAIN.body if isinstance(n,ast.Try)).body


def assignment(node, name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),
                 'actual_runner_ast','exec'),namespace)


def input_contract(config, workload, warmups):
    start=next(i for i,n in enumerate(BODY) if assignment(n,'lengths'))
    end=next(i for i,n in enumerate(BODY) if isinstance(n,ast.Expr)
             and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
             and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='config'
             and n.value.func.attr=='update')
    scope=dict(config=copy.deepcopy(config),workload=copy.deepcopy(workload),warmups=warmups)
    execute(BODY[start:end],scope)
    return scope


def prepared(name):
    folder=ROOT.parent/'healthy_workload/prepared'/name
    config,work=RUNNER.load_inputs(folder/'inputs')
    warmups={d:RUNNER.load_inputs(folder/'warmups'/d) for d in ('short','long')}
    return config,work,warmups


def fake_engine(usable, spec_type):
    pool=NS(num_gpu_blocks=usable+1,get_num_free_blocks=lambda:usable,
            null_block=NS(block_id=0,is_null=True))
    spec=spec_type();spec.block_size=16;spec.sliding_window=None;spec.attention_chunk_size=None
    group=NS(kv_cache_spec=spec,is_eagle_group=False,layer_names=['fixture'])
    layout=NS(kv_cache_groups=[group],num_blocks=usable+1)
    coordinator=type('KVCacheCoordinatorNoPrefixCache',(),{})()
    coordinator.single_type_managers=[NS(block_pool=pool,block_size=16)]
    coordinator.scheduler_block_size=16
    manager=NS(block_pool=pool,coordinator=coordinator,num_kv_cache_groups=1,watermark_blocks=0,
               enable_caching=False,use_eagle=False,kv_cache_config=layout)
    scheduler=NS(kv_cache_manager=manager,kv_cache_config=layout,requests={},
                 get_request_counts=lambda:(0,0),num_lookahead_tokens=0,num_spec_tokens=0,
                 dcp_world_size=1,pcp_world_size=1)
    return NS(engine_core=NS(engine_core=NS(scheduler=scheduler)),has_unfinished_requests=lambda:False,
              vllm_config=NS(cache_config=NS(enable_prefix_caching=False),speculative_config=None,
                             scheduler_config=NS(max_num_seqs=32),model_config=NS(max_model_len=4096)))


class HealthyRunnerTests(unittest.TestCase):
    def test_exact_default_and_alternate_input_contracts(self):
        for name,blocks in [('dev16_burst_kv768',768),('dev16_burst_kv4096',4096),('holdout16_steady_kv768',768)]:
            config,work,warmups=prepared(name)
            scope=input_contract(config,work,warmups)
            self.assertEqual(scope['usable_blocks'],blocks)
            self.assertEqual(scope['kv_bytes'],(blocks+1)*2097152)
            self.assertEqual(config['requests'],16)
            self.assertEqual(config['model']['revision'],'7f1c97f440f06ce36705e4f2b843edb5925f4498')
        default=ROOT.parent/'healthy_workload/prepared/dev16_burst_kv768'
        for path in ('inputs/config.json','inputs/workload.json','warmups/short/config.json',
                     'warmups/short/workload.json','warmups/long/config.json','warmups/long/workload.json'):
            self.assertEqual((PKG/path).read_bytes(),(default/path).read_bytes())
        config,work,warmups=prepared('dev16_burst_kv768')
        config['fixed_kv_cache_memory_bytes']+=2097152
        with self.assertRaises(ValueError):input_contract(config,work,warmups)
        config,work,warmups=prepared('dev16_burst_kv768')
        work['sampling']['ignore_eos']=True
        with self.assertRaises(ValueError):input_contract(config,work,warmups)

    def test_actual_constructor_kwargs_keep_full32_and_dynamic_kv(self):
        start=next(i for i,n in enumerate(BODY) if assignment(n,'kwargs'))
        for name,blocks in [('dev16_burst_kv768',768),('dev16_burst_kv4096',4096)]:
            config,work,warmups=prepared(name);scope=input_contract(config,work,warmups)
            scope['model']=config['model']
            execute(BODY[start:start+2],scope)
            kwargs=scope['kwargs']
            self.assertEqual(kwargs['kv_cache_memory_bytes'],(blocks+1)*2097152)
            self.assertEqual(kwargs['max_num_seqs'],32)
            self.assertEqual(kwargs['max_num_batched_tokens'],1024)
            self.assertEqual(kwargs['kv_offloading_size'],16)
            self.assertEqual(kwargs['revision'],config['model']['revision'])
            self.assertTrue(kwargs['scheduler_reserve_full_isl'])

    def test_actual_safe_cap_and_live_allocation_predicates(self):
        spec_type=type('FullAttentionSpec',(),{})
        vllm=ModuleType('vllm');v1=ModuleType('vllm.v1');interface=ModuleType('vllm.v1.kv_cache_interface')
        interface.FullAttentionSpec=spec_type
        checks=[n for n in BODY if isinstance(n,ast.If) and any(isinstance(x,ast.Name)
                and x.id in ('host_init','qualification') for x in ast.walk(n.test))]
        self.assertEqual(len(checks),2)
        with patch.dict(sys.modules,{'vllm':vllm,'vllm.v1':v1,'vllm.v1.kv_cache_interface':interface}):
            for blocks in (768,4096):
                config,_,_=prepared('dev16_burst_kv768')
                result=SAFE.qualify_safe_cap(fake_engine(blocks,spec_type),config)
                self.assertEqual(result['status'],'QUALIFIED')
                self.assertEqual(result['usable_blocks'],blocks)
                self.assertEqual(result['safe_cap'],min(32,blocks//80))
                result['observed_scheduler_reserve_full_isl']=True
                scope=dict(host_init={'cpu_kv':{'unique_storage_bytes':16*1024**3},'manager':{'capacity_blocks':8192}},
                           qualification=result,usable_blocks=blocks,kv_bytes=(blocks+1)*2097152,
                           memory={'kv_storage_bytes':(blocks+1)*2097152})
                execute(checks,scope)
                scope['memory']['kv_storage_bytes']-=2097152
                with self.assertRaises(RuntimeError):execute(checks,scope)
                scope['memory']['kv_storage_bytes']+=2097152
                scope['host_init']['manager']['capacity_blocks']=8191
                with self.assertRaises(RuntimeError):execute(checks,scope)

    def test_real_capture_code_retains_finish_and_native_stop_identifier(self):
        vllm=ModuleType('vllm');sampling=ModuleType('vllm.sampling_params')
        sampling.RequestOutputKind=NS(CUMULATIVE='fixture')
        params=[]
        def SamplingParams(**kwargs):params.append(kwargs);return NS(**kwargs)
        vllm.SamplingParams=SamplingParams
        class Engine:
            active=False
            vllm_config=NS(scheduler_config=NS(async_scheduling=False,stream_interval=1))
            def has_unfinished_requests(self):return self.active
            def add_request(self,rid,prompt,params,arrival_time):self.active=True;self.rid=rid;return rid+'-internal'
            def step(self):
                self.active=False
                return [NS(request_id=self.rid,finished=True,outputs=[NS(token_ids=[123],finish_reason='stop',stop_reason=50279)])]
        config,work,_=prepared('dev16_burst_kv768')
        work=dict(work,source_requests=work['source_requests'][:1],actual_prompt_token_ids=work['actual_prompt_token_ids'][:1],arrival_traces_s={'steady':[0.0]})
        with patch.dict(sys.modules,{'vllm':vllm,'vllm.sampling_params':sampling}):
            raw=MEASURE.measure_episode(Engine(),work,config,'steady',1.,'fixture')
        self.assertEqual(raw['status'],'COMPLETE')
        self.assertEqual(raw['requests'][0]['finish_reason'],'stop')
        self.assertEqual(raw['requests'][0]['native_stop_reason'],50279)
        self.assertEqual(params[0]['max_tokens'],512)
        self.assertEqual(params[0]['stop'],[])
        self.assertEqual(params[0]['stop_token_ids'],[])
        self.assertFalse(params[0]['ignore_eos'])


if __name__=='__main__':
    unittest.main()
