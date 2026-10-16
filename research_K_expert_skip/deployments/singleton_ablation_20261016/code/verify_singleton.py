#!/usr/bin/env python3
"""Focused ablation checks plus the unchanged legacy suite; CUDA is opt-in."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import unittest

import expert_skip
from expert_skip import ExpertSkipPolicy, SkipConfig, reference_policy
import compile_policy_offline as offline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HAS_TORCH = importlib.util.find_spec('torch') is not None
CUDA = False
WEIGHTS = [[1.,4.,1.,2.], [1.,4.,1.,2.]]
IDS = [[0,1,2,3], [4,5,2,6]]


def config(require=True, **kw):
    return SkipConfig(**{'mode':'marginal','tail_rank':2,'gate_threshold':.125,
                         'mass_budget':.25,'require_singleton':require, **kw})


class ReferenceChecks(unittest.TestCase):
    def test_default_compatibility_and_strict_boolean(self):
        self.assertTrue(SkipConfig().require_singleton)
        for value in (None,0,1,'false'):
            with self.assertRaises(ValueError):
                config(value)
        legacy = {k:v for k,v in asdict(config()).items() if k!='require_singleton'}
        self.assertEqual(reference_policy(WEIGHTS,IDS,7,SkipConfig(**legacy)),
                         reference_policy(WEIGHTS,IDS,7,config(True)))

    def test_only_count_predicate_and_inclusive_boundaries(self):
        _,on,on_stats = reference_policy(WEIGHTS,IDS,7,config(True),False,100)
        off_w,off,off_stats = reference_policy(WEIGHTS,IDS,7,config(False),False,100)
        self.assertEqual(on,[[-1,1,2,3],[-1,5,2,6]])
        self.assertEqual(off,[[-1,1,-1,3],[-1,5,-1,6]])
        self.assertEqual(off_w,[[0.,4.,0.,2.]]*2)
        self.assertEqual(on_stats['skipped_slots'],2)
        self.assertEqual(off_stats['skipped_slots'],4)
        self.assertEqual(off_stats['logical_weight_bytes_before']-off_stats['logical_weight_bytes_after'],300)
        self.assertEqual(off_stats['removed_gate_mass_max'],.25)
        _,below,_ = reference_policy(WEIGHTS,IDS,7,config(False,gate_threshold=.125-2**-20),False)
        self.assertEqual(below,IDS)
        _,limited,_ = reference_policy(WEIGHTS,IDS,7,config(False,mass_budget=.25-2**-20),False)
        self.assertEqual(limited,[[0,1,-1,3],[4,5,-1,6]])

    def test_scaled_renorm_and_at_least_one_expert(self):
        for require in (True,False):
            w,_,_ = reference_policy(WEIGHTS,IDS,7,config(require),True)
            self.assertTrue(all(math.isclose(sum(row),8.) for row in w))
            w,ids,_ = reference_policy([[0.,0.,0.]]*2,[[0,1,2]]*2,3,
                                      config(require,tail_rank=0,mass_budget=0.),True)
            self.assertTrue(all(any(x>=0 for x in row) for row in ids))
            self.assertTrue(all(math.isfinite(x) for row in w for x in row))
        self.assertEqual(reference_policy([],[],7,config(False))[2]['tokens'],0)

    def test_other_modes_and_t1_half_open_boundary_unchanged(self):
        for mode in ('dense','global','gate','rank_gate'):
            kw={'mode':mode,'tail_rank':2,'rank_end':4,'gate_threshold':.125,'mass_budget':.25}
            a=reference_policy(WEIGHTS,IDS,7,SkipConfig(**kw,require_singleton=True))
            b=reference_policy(WEIGHTS,IDS,7,SkipConfig(**kw,require_singleton=False))
            self.assertEqual(a,b)
            if mode=='rank_gate':
                self.assertEqual(a[1],IDS)  # g==upper remains excluded in T1.

    def test_arm_pairs_differ_only_in_boolean(self):
        arms=json.loads((HERE.parent/'arms.json').read_text())['arms']
        self.assertEqual(len(arms),8)
        for i in (2,4,6):
            on,off=arms[i]['spec'],arms[i+1]['spec']
            self.assertTrue(on['require_singleton'])
            self.assertFalse(off['require_singleton'])
            self.assertEqual({**off,'require_singleton':True},on)
            SkipConfig(**on);SkipConfig(**off)
        full=json.loads((HERE.parent/'arms_task_quality.json').read_text())['arms']
        self.assertEqual(len(full),14)
        self.assertEqual(len({row['name'] for row in full}),14)
        for row in full:
            if row['spec'] and row['spec']['mode']!='global_native':
                SkipConfig(**row['spec'])

    def test_offline_kernel_signature_includes_boolean(self):
        tree=ast.parse((HERE/'expert_skip.py').read_text())
        functions={node.name:node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
        for case in offline.cases([0,4],[(64,6),(64,8)],['marginal','rank_gate'],[False,True],['fp32'],[True,False]):
            for name,pointers,constants,_ in offline.kernel_specs(case):
                self.assertEqual(set(pointers)|set(constants),{arg.arg for arg in functions[name].args.args})
                if name in ('policy_kernel','fused_small_kernel'):
                    self.assertEqual(constants['REQUIRE_SINGLETON'],case['require_singleton'])


@unittest.skipUnless(HAS_TORCH,'torch not installed')
class TorchChecks(unittest.TestCase):
    def compare(self, weights, ids, cfg, renorm, active=None, stats=True, backend='torch', device='cpu'):
        import torch
        policy=ExpertSkipPolicy(SkipConfig(**{**asdict(cfg),'backend':backend}),collect_stats=stats)
        w=torch.tensor(weights,dtype=torch.float32,device=device)
        i=torch.tensor(ids,dtype=torch.int32,device=device)
        a=None if active is None else torch.tensor(active,dtype=torch.int32,device=device)
        got_w,got_i=policy('layer',w,i,64,renorm,100,active_tokens=a)
        real=len(ids) if active is None else max(0,min(active,len(ids)))
        expected_w,expected_i,expected_s=reference_policy(weights[:real],ids[:real],64,cfg,renorm,100)
        self.assertEqual(got_i.tolist(),expected_i+ids[real:])
        torch.testing.assert_close(got_w,torch.tensor(expected_w+weights[real:],device=device),rtol=2e-6,atol=1e-7)
        if stats:
            expected_s['events']=1 if active is None else int(real>0)
            actual=policy.snapshot()['totals']
            for key,value in expected_s.items():
                self.assertAlmostEqual(actual[key],value,delta=max(1e-5,abs(value)*1e-6))
        else:
            self.assertEqual(policy.snapshot()['layers'],{})

    def test_reference_torch_boundaries_padding_and_zero_active(self):
        for require in (True,False):
            for renorm in (False,True):
                for stats in (False,True):
                    for active in (None,0,1,2):
                        self.compare(WEIGHTS*2,IDS*2,config(require),renorm,active,stats)
                    for budget in (.25,.25-2**-20):
                        self.compare(WEIGHTS,IDS,config(require,mass_budget=budget),renorm,stats=stats)

    def test_no_input_preserves_shape_and_legacy_event_semantics(self):
        import torch
        for require in (True,False):
            for active in (None,torch.tensor(0,dtype=torch.int32)):
                p=ExpertSkipPolicy(config(require))
                w,i=p('empty',torch.empty((0,8)),torch.empty((0,8),dtype=torch.int32),64,False,active_tokens=active)
                self.assertEqual(tuple(w.shape),(0,8));self.assertEqual(tuple(i.shape),(0,8))
                self.assertEqual(p.snapshot()['totals']['events'],int(active is None))
                self.assertEqual(p.snapshot()['totals']['tokens'],0)

    def test_unsorted_k6_k8_reference_parity(self):
        rng=random.Random(918)
        for batch,k in ((1,6),(8,8),(17,6)):
            ids=[rng.sample(range(64),k) for _ in range(batch)]
            weights=[[float(rng.randrange(1,100)) for _ in range(k)] for _ in range(batch)]
            for require in (True,False):
                for renorm in (False,True):
                    self.compare(weights,ids,config(require,tail_rank=4,gate_threshold=.15,mass_budget=.3),renorm)


class CudaChecks(TorchChecks):
    # Only the focused CUDA methods are loaded below, not the inherited CPU tests.
    def test_cuda_boundary_parity(self):
        for backend in ('triton','triton_multi'):
            for require in (True,False):
                for renorm in (False,True):
                    for stats in (False,True):
                        self.compare(WEIGHTS*2,IDS*2,config(require),renorm,2,stats,backend,'cuda')
                        self.compare(WEIGHTS,IDS,config(require,mass_budget=.25-2**-20),renorm,None,stats,backend,'cuda')

    def test_cuda_graph_replay_zero_active_and_empty(self):
        import torch
        for backend in ('triton','triton_multi'):
            for require in (True,False):
                cfg=SkipConfig(**{**asdict(config(require)),'backend':backend})
                p=ExpertSkipPolicy(cfg)
                weights=torch.tensor(WEIGHTS*2,device='cuda');ids=torch.tensor(IDS*2,device='cuda',dtype=torch.int32)
                active=torch.tensor(2,device='cuda',dtype=torch.int32)
                def call():return p('layer',weights,ids,64,True,100,active_tokens=active)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):call()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):w,i=call()
                for actual in (2,1,0,4):
                    p.reset();active.fill_(actual);graph.replay()
                    expected_w,expected_i,stats=reference_policy((WEIGHTS*2)[:actual],(IDS*2)[:actual],64,cfg,True,100)
                    self.assertEqual(i.tolist(),expected_i+(IDS*2)[actual:])
                    torch.testing.assert_close(w,torch.tensor(expected_w+(WEIGHTS*2)[actual:],device='cuda'))
                    self.assertEqual(p.snapshot()['totals']['tokens'],actual)
                    self.assertEqual(p.snapshot()['totals']['events'],int(actual>0))
                for scalar in (None,torch.tensor(0,device='cuda',dtype=torch.int32)):
                    p.reset()
                    w,i=p('layer',torch.empty((0,8),device='cuda'),torch.empty((0,8),device='cuda',dtype=torch.int32),64,False,active_tokens=scalar)
                    self.assertEqual(tuple(w.shape),(0,8));self.assertEqual(tuple(i.shape),(0,8))
                    self.assertEqual(p.snapshot()['totals']['events'],int(scalar is None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda',action='store_true',help='requires the owner-held GPU lock')
    args=parser.parse_args()
    if Path(expert_skip.__file__).resolve()!=HERE/'expert_skip.py':
        raise RuntimeError('Tests imported the wrong core; refusing misleading success')
    os.environ['EXPERT_SKIP_TEST_CUDA']='1' if args.cuda else '0'
    # Import payload first: the unchanged legacy file prepends root/code, but
    # its from expert_skip import resolves the already-verified payload module.
    legacy_path=ROOT/'code/tests/test_expert_skip.py'
    spec=importlib.util.spec_from_file_location('legacy_policy_tests',legacy_path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    if legacy.SkipConfig is not SkipConfig:
        raise RuntimeError('Legacy suite did not resolve the isolated payload')
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([loader.loadTestsFromTestCase(ReferenceChecks),loader.loadTestsFromTestCase(TorchChecks),loader.loadTestsFromModule(legacy)])
    if args.cuda:
        suite.addTests(CudaChecks(name) for name in ('test_cuda_boundary_parity','test_cuda_graph_replay_zero_active_and_empty'))
    print(json.dumps({'core':str(Path(expert_skip.__file__).resolve()),'core_sha256':hashlib.sha256((HERE/'expert_skip.py').read_bytes()).hexdigest(),
                      'cuda_requested':args.cuda,'scope':'CUDA execution' if args.cuda else 'CPU-only; CUDA not initialized or tested'}),flush=True)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
