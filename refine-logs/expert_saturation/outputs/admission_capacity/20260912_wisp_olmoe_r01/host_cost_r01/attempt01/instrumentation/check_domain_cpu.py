"""Rerun the existing original-helper argument and restoration check without GPU libraries."""
import ast
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types

HERE = Path(__file__).resolve().parent
path = HERE / 'compile_domain.py'
spec = importlib.util.spec_from_file_location('compile_domain_check', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fused = HERE / 'installed_source' / 'fused_moe.py'
align = HERE / 'installed_source' / 'moe_align_block_size.py'
fixture_hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fused, align)}
assert fixture_hashes == mod.EXPECTED
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wisp-source', type=Path, default=HERE / 'installed_source' / 'wisp_fused_moe.py')
wisp_source = parser.parse_args().wisp_source
wisp_hash = hashlib.sha256(wisp_source.read_bytes()).hexdigest()
assert wisp_hash == '5572d4f05593a5a9fc4adaa14421cc596886a435b6f5f51f476a1dae6e521857'
state_class = next(n for n in ast.parse(wisp_source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'WispMoEState')
state_slots = ast.literal_eval(next(n.value for n in state_class.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == '__slots__' for t in n.targets)))
CPUState = type('WispMoEStateCPU', (), {'__slots__': state_slots})
stat_names = {name for name in state_slots if name.startswith('stats_')}


class Tensor:
    def __init__(self, shape, dtype='bf16', pointer=None):
        self.shape = tuple(shape); self.dtype = dtype; self.device = 'cuda'; self.pointer = pointer or id(self)
    def __getitem__(self, s): return Tensor(((s.stop or self.shape[0])-(s.start or 0), *self.shape[1:]), self.dtype, self.pointer)
    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype, self.pointer)
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 2 if self.dtype == 'bf16' else 4
    def data_ptr(self): return self.pointer
    def is_contiguous(self): return True
    def size(self, i=None): return self.shape if i is None else self.shape[i]
    def stride(self, i): return math.prod(self.shape[i+1:])


class Kernel:
    def __init__(self, fail=None): self.calls = []; self.fail = fail
    def __getitem__(self, grid): return lambda *a, **kw: self.run(*a, grid=grid, warmup=False, **kw)
    def run(self, *a, **kw):
        assert kw.pop('warmup') is True, 'would execute GPU kernel'
        kw.pop('grid'); m = a[2].shape[0]; gemm = 2 if kw['MUL_ROUTED_WEIGHT'] else 1
        assert a[6].dtype == 'fp32' and a[6].shape == (m, 8)
        assert a[0].shape == ((m, 2048) if gemm == 1 else (m*8, 1024))
        assert a[2].shape == (m, 8, 2048) and kw['top_k'] == (8 if gemm == 1 else 1)
        assert a[13] == m*8
        self.calls.append((m, gemm, a[1].shape[0], a[12], kw['naive_block_assignment'], kw))
        if len(self.calls) == self.fail: raise RuntimeError('injected compile failure')
        src = types.SimpleNamespace(signature={'topk_weights_ptr': '*fp32'}, constants=kw, attrs={'em_mod16': a[12] % 16})
        c = types.SimpleNamespace(hash=hashlib.sha256(json.dumps([kw, src.attrs], sort_keys=True).encode()).hexdigest(), src=src, module=None)
        c._init_handles = lambda: setattr(c, 'module', 1)
        return c


cuda = types.SimpleNamespace(**{n: lambda device: 0 for n in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved')})
torch = types.SimpleNamespace(bfloat16='bf16', float32='fp32', int32='i32', cuda=cuda,
                              empty=lambda n, dtype, device: Tensor((n,), dtype))


def native_function(name, namespace):
    node = next(n for n in ast.parse(fused.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    for a in node.args.args + node.args.kwonlyargs: a.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(fused), 'exec'), namespace)
    return namespace[name]


ns = {'envs': types.SimpleNamespace(VLLM_BATCH_INVARIANT=False), 'current_platform': types.SimpleNamespace(is_rocm=lambda: False)}
default = native_function('get_default_config', ns)
original_import = mod.importlib.import_module
try:
    with tempfile.TemporaryDirectory() as tmp:
        results = []
        for mode, fail in [('fullstage', None), ('oneshot', None), ('oneshot', 13)]:
            kernel = Kernel(fail)
            namespace = {'fused_moe_kernel': kernel, 'triton': types.SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b)}
            invoke = native_function('invoke_fused_moe_triton_kernel', namespace)
            moe = types.SimpleNamespace(__file__=str(fused), fused_moe_kernel=kernel, tl=types.SimpleNamespace(bfloat16='bf16'),
                invoke_fused_moe_triton_kernel=invoke, try_get_optimal_moe_config=lambda w1, w2, k, d, m: default(m, w2[0], w2[2], w1[2], k, d))
            mod.importlib.import_module = lambda name: moe if name.endswith('.fused_moe') else types.SimpleNamespace(__file__=str(align))
            pool = (Tensor((384, 2048, 2048)), Tensor((384, 2048, 1024)))
            cap = 20 if mode == 'fullstage' else 21
            layers = {}
            for i in range(16):
                values = dict(layer_idx=i, num_experts=64, mode='paged', cap_experts=cap,
                    slot_to_expert=[-1]*cap, expert_to_slot={}, lru_tick=[0]*cap, lru_clock=0,
                    last_topk_ids_cpu=None, last_unique_experts=set(), scratch_w13=pool[0][i*cap:(i+1)*cap],
                    scratch_w2=pool[1][i*cap:(i+1)*cap], expert_map_device=Tensor((64,), 'i32'),
                    cpu_w13=Tensor((64, 2048, 2048)), cpu_w2=Tensor((64, 2048, 1024)), copy_stream=None, copy_event=None)
                values.update(dict.fromkeys(stat_names, 0))
                assert set(values) == set(state_slots)
                state = CPUState()
                for name, value in values.items(): setattr(state, name, value)
                assert not hasattr(state, '__dict__')
                try: vars(state)
                except TypeError: pass
                else: raise AssertionError('CPU state must reproduce the legacy vars(state) failure')
                layers[i] = dict(layer_name=f'model.layers.{i}.mlp.experts', state=state, ever_loaded=set())
            runtime = types.SimpleNamespace(torch=torch, shared_mode=mode, shared_weights=pool, layers=layers, records=[], events=[],
                flushed_calls=0, context={'phase': 'compile_domain_warmup'}, measurement=False)
            output = Path(tmp) / (mode + str(fail) + '.json')
            try: report = mod.precompile(runtime, mode, output)
            except RuntimeError:
                assert fail is not None
                report = json.loads(output.read_text())
            assert 'run' not in vars(kernel) and report['state_equal']
            assert all(set(layer['stats']) == stat_names for layer in report['before']['layers'].values())
            assert report['actual_invocations'] == (fail or 320)
            assert [(c[0], c[1]) for c in kernel.calls] == [(m, g) for m in range(1, 161) for g in (1, 2)][:fail]
            assert all(c[4] == (mode == 'fullstage' and c[0] <= 2) for c in kernel.calls)
            if fail: assert report['status'] == 'FAILED' and report['calls'][-1]['status'] == 'FAILED'
            else: assert report['status'] == 'COMPLETE_COMPILE_AND_LOAD_ONLY'
            results.append(dict(mode=mode, injected_failure=fail, invocations=len(kernel.calls), state_equal=report['state_equal'], restored=True))
        print(json.dumps(dict(checks='original invoke helper AST + original default config, CPU tensors only',
            source_fixture_sha256=fixture_hashes, wisp_source_sha256=wisp_hash, state_slots=list(state_slots),
            legacy_vars_raises_typeerror=True, compile_domain_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), results=results), indent=2))
finally:
    mod.importlib.import_module = original_import
