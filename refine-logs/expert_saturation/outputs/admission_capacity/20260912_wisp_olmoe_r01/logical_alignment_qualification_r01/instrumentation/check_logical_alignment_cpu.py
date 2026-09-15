"""CPU interface fixture using the hash-checked native assignment function AST."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--fused-source', type=Path, default=ROOT.parent / 'compile_domain_r02/instrumentation/installed_source/fused_moe.py')
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('logical_check', ROOT / 'instrumentation/run_logical_alignment.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
assert probe.digest(args.fused_source) == probe.EXPECTED['fused_moe']
tree = ast.parse(args.fused_source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_expert_assignment')
observed, fail_alignment = [], False
def align(ids, block, experts, mapping, **kw):
    observed.append((ids.shape[0], experts, mapping.shape, kw['ignore_invalid_experts']))
    if fail_alignment: raise ValueError('alignment failure fixture')
    return 'assignments'
namespace = dict(moe_align_block_size=align)
exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
native = namespace['_prepare_expert_assignment']
module = SimpleNamespace(_prepare_expert_assignment=native)

class Tensor:
    def __init__(self, shape, data=None): self.shape, self.data, self.device = shape, data, 'cpu-fixture'
    def __getitem__(self, part):
        return Tensor((len(range(self.shape[0])[part]), *self.shape[1:]), None if self.data is None else self.data[part])
    def contiguous(self): return self

sys.path.insert(0, str(ROOT / 'source'))
from shared_pool_plan import plan_shared_pool
from analyze_layer_budget import LRU
plan = plan_shared_pool(list(range(64)), LRU(21).snapshot(), 0)
mapping = plan['expert_map_device']
kwargs = dict(hidden_states=Tensor((160, 2048)), w1=Tensor((384, 2048, 2048)), w2=Tensor((384, 2048, 1024)),
              topk_ids=Tensor((160, 8)), topk_weights=Tensor((160, 8)), expert_map=Tensor((384,), mapping), global_num_experts=384)
def kernel(**kw):
    if kw['expert_map'] is None: return 'E64 untouched'
    module._prepare_expert_assignment(kw['topk_ids'], {'BLOCK_SIZE_M': 64}, kw['hidden_states'].shape[0], 8,
        kw['global_num_experts'], kw['expert_map'], ignore_invalid_experts=True)
    return 'bad' if kw['expert_map'].data[0] != mapping[0] else ('logical' if kw['global_num_experts'] == 64 else 'reference')
record = dict(call_id=0, layer_name='model.layers.0.mlp.experts', context=dict(phase='measurement', step_id=0), rows=160,
              row_topk_experts=[[((i * 8) + j) % 64 for j in range(8)] for i in range(160)], active_experts=list(range(64)), shared_plan=plan)
runtime = SimpleNamespace(kernel=kernel, records=[record], shared_mode='oneshot', measurement=False, validation_enabled=False,
    shared_weights=(kwargs['w1'], kwargs['w2']), torch=SimpleNamespace(int32='int32', tensor=lambda v, **kw: Tensor((len(v),), v)))
report = dict(actual_calls=[], prefix_cases=[], negative_control=None)
probe.metrics = lambda torch, actual, ref: dict(finite=True, allclose=actual != 'bad', bit_equal=actual != 'bad', maxabs=0 if actual != 'bad' else 1,
                                               reference_l2=1, relative_l2=0 if actual != 'bad' else 1, atol=.01, rtol=.01)
hook = probe.Qualification(runtime, module, report)
assert runtime.kernel(**kwargs) == 'reference' and not report['actual_calls']
runtime.measurement = True; observed.clear()
assert runtime.kernel(**kwargs) == 'logical'
assert len(observed) == 13 and module._prepare_expert_assignment is native
assert [p['rows'] for p in report['prefix_cases']] == list(probe.WIDTHS)
assert report['prefix_cases'][-1]['reused_actual_comparison'] and hook.covered[record['layer_name']] == list(probe.WIDTHS)
assert report['negative_control']['status'] == 'complete'
assert all(g == (384 if flag else 64) and shape == (g,) for _, g, shape, flag in observed)
assert runtime.kernel(**dict(kwargs, global_num_experts=64, expert_map=None)) == 'E64 untouched'
for bad in ('alignment_exception', 'missing_hook'):
    fail_alignment = bad == 'alignment_exception'; case = {}
    try:
        probe.invoke(module, kernel if fail_alignment else lambda **kw: None, kwargs, False, case, bad)
    except (ValueError, RuntimeError): pass
    else: raise AssertionError('failure did not propagate')
    assert module._prepare_expert_assignment is native and case['assignment_restored']
fail_alignment = False
invalid = dict(record, row_topk_experts=[[0] * 8] * 160)
for bad in (invalid, dict(record, shared_plan=dict(plan, expert_map_device=[384, *mapping[1:]])),
            dict(record, shared_plan=dict(plan, expert_map_device=[mapping[1], *mapping[1:]]))):
    try: probe.validate_record(bad, kwargs)
    except ValueError: pass
    else: raise AssertionError('invalid CPU route/map accepted')
compile((ROOT / 'instrumentation/run_logical_alignment.py').read_text(), 'run_logical_alignment.py', 'exec')
assert not any(name in sys.modules for name in ('torch', 'vllm'))
print(json.dumps(dict(status='PASS', scope='CPU argument/restore/prefix fixture; numerical metrics stubbed, no CUDA',
    native_helper_sha256=probe.digest(args.fused_source), actual_and_prefix_and_negative_invocations=13,
    real_plan=True, actual160_not_repeated=True, warmup_and_E64_passthrough=True, failure_restore=True)))
