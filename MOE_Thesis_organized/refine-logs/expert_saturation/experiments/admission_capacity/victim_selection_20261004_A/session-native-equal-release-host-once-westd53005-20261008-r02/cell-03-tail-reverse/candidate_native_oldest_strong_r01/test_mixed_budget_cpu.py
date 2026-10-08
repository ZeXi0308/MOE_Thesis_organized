"""Check budget preservation at the runner boundary and real performance sampler path."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

PKG = Path(__file__).parent / 'pkg'
sys.path.insert(0, str(PKG))
from request_measurement import measure_episode
from run_recovery_cadence import validated_output_overrides


def workload():
    return dict(source_requests=[dict(request_id=rid, document_id=rid) for rid in ('short','long')],
        actual_prompt_token_ids=[[10,11],[12,13]], arrival_traces_s={'steady':[0.,0.]})


class FakeEngine:
    def __init__(self):
        self.vllm_config=NS(scheduler_config=NS(async_scheduling=False,stream_interval=1))
        self.params={}
        self.pending=[]

    def add_request(self, rid, prompt, params, arrival_time):
        self.params[rid]=params
        self.pending.append(rid)
        return rid+'-internal'

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        pending,self.pending=self.pending,[]
        # A natural early stop verifies that assigned caps do not force lengths.
        return [NS(request_id=rid,finished=True,
                   outputs=[NS(token_ids=[42],finish_reason='stop')]) for rid in pending]


class MixedBudgetTests(unittest.TestCase):
    def test_runner_preserves_map_and_performance_path_uses_assigned_caps(self):
        config=dict(output_tokens=1024,output_tokens_by_request={'short':128},ignore_eos=False,min_tokens=0)
        output_overrides=validated_output_overrides(config,workload())
        tree=ast.parse((PKG/'run_recovery_cadence.py').read_text())
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        # Execute the actual config.update RHS: this catches reintroducing {}.
        update=next(node for node in ast.walk(main) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and node.func.attr=='update'
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='config'
            and any(kw.arg=='output_tokens_by_request' for kw in node.keywords))
        rhs=next(kw.value for kw in update.keywords if kw.arg=='output_tokens_by_request')
        config.update(output_tokens_by_request=eval(compile(ast.Expression(rhs),'<runner-budget-rhs>','eval'),
            {'output_overrides':output_overrides}))
        self.assertEqual(config['output_tokens_by_request'],{'short':128})
        warm=next(node for node in ast.walk(main) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='warm_config' for t in node.targets))
        warm_rhs=next(kw.value for kw in warm.value.keywords if kw.arg=='output_tokens_by_request')
        self.assertEqual(ast.literal_eval(warm_rhs),{})
        vllm=ModuleType('vllm');vllm.SamplingParams=lambda **kwargs:NS(**kwargs)
        sampling=ModuleType('vllm.sampling_params');sampling.RequestOutputKind=NS(CUMULATIVE='cumulative')
        engine=FakeEngine()
        with patch.dict(sys.modules,{'vllm':vllm,'vllm.sampling_params':sampling}):
            result=measure_episode(engine,workload(),config,'steady',1.,'cpu-fixture')
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual([r['max_output_tokens'] for r in result['requests']],[128,1024])
        self.assertEqual([engine.params['cpu-fixture/'+rid].max_tokens for rid in ('short','long')],[128,1024])
        self.assertTrue(all(not p.ignore_eos and p.min_tokens==0 for p in engine.params.values()))
        self.assertEqual(validated_output_overrides({'output_tokens':1024},workload()),{})

    def test_unknown_ids_invalid_caps_and_context_are_rejected(self):
        for overrides in ({'unknown':128},{'short':1},{'short':1025},{'short':True},{'short':128.0},None):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validated_output_overrides(dict(output_tokens=1024,output_tokens_by_request=overrides),workload())
        for maximum in (1,1025,True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                validated_output_overrides(dict(output_tokens=maximum),workload())
        too_long=workload();too_long['actual_prompt_token_ids'][0]=[1]*4000
        with self.assertRaises(ValueError):
            validated_output_overrides(dict(output_tokens=1024,output_tokens_by_request={'short':128}),too_long)
        self.assertEqual(validated_output_overrides(
            dict(output_tokens=1024,output_tokens_by_request={'short':96}),too_long),{'short':96})


if __name__ == '__main__':
    unittest.main()
