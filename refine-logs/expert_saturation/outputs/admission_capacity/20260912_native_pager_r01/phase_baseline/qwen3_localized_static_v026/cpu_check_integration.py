"""CPU-only scope/guard checks; fake tensors do not validate numerical kernels."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "source" / (name + ".py"))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


scope, adapter = module("localized_scope"), module("wisp_v026_adapter")


class Tensor:
    counter = 0

    def __init__(self):
        Tensor.counter += 1
        self.pointer = Tensor.counter; self.device = "cuda:0"; self.dtype = "bfloat16"; self.shape = [1, 2]

    def data_ptr(self): return self.pointer
    def untyped_storage(self): return self
    def nbytes(self): return 4
    def numel(self): return 2
    def element_size(self): return 2


def fixture(path):
    cuda = NS(synchronize=lambda: None)
    runtime = adapter._Runtime(48, path, NS(cuda=cuda), NS(__file__=__file__), lambda **kw: None)
    for i in range(48):
        state = NS(**{key: Tensor() for key in ("cpu_w13", "cpu_w2", "scratch_w13", "scratch_w2", "expert_map_device")},
                   cap_experts=48, num_experts=128, slot_to_expert=[-1] * 48, lru_tick=[0] * 48, lru_clock=0)
        name = f"model.layers.{i}.mlp.experts"
        layer = NS(layer_name=name, w13_weight=Tensor(), w2_weight=Tensor())
        runtime.layers[i] = dict(layer=layer, state=state, layer_name=name, ever_loaded={i}, validated=True)
    runtime.apply = MethodType(lambda self: (id(self), self.validation_enabled, self.next_call_id), runtime)
    runtime.group_retention = "none"; runtime.group_retention_order = "early"; runtime.retention_validated = False
    runtime.memory_observer_mode = "nested"; runtime.numerical_localization = object()
    model = NS(named_parameters=lambda: [(e["layer_name"], e["layer"].w13_weight) for e in runtime.layers.values()])
    runner = NS(model=model, kv_caches=[Tensor() for _ in range(48)])
    scheduler = NS(requests={}, running=[], waiting=[])
    engine = NS(has_unfinished_requests=lambda: False, engine_core=NS(engine_core=NS(scheduler=scheduler)))
    return runtime, engine, runner


class Integration(unittest.TestCase):
    def test_gate_keeps_original_failed_allclose(self):
        _, original = scope.read_qualification(ROOT / "qualification_prepared")
        aliases = {f"internal{i}": rid for i, rid in enumerate(dict.fromkeys(r["request_id"] for r in original["rows"]))}
        inverse = {v: k for k, v in aliases.items()}
        names = [f"model.layers.{i}.mlp.experts" for i in range(48)]
        values = [dict(original["diagnostic"], layer_name=name, rows=32, call_id=i, allclose=i != 47) for i, name in enumerate(names)]
        record = dict(call_id=47, rows=32, context=dict(rows=[dict(internal_request_id=inverse[r["request_id"]],
            computed_position=r["computed_position"], prompt_tokens=r["prompt_tokens"]) for r in original["rows"]]),
            row_topk_experts=original["row_topk_experts"], groups=[dict(required_experts=g) for g in original["required_experts"]])
        runtime = NS(validation_results=values, layers={i: dict(layer_name=n) for i, n in enumerate(names)}, records=[record])
        raw = dict(status="COMPLETE", internal_to_source=aliases, requests=[dict(status="completed", output_token_ids=list(range(8))) for _ in range(4)])
        probe = NS(gate=lambda: dict(status="PASS", complete=True))
        passed = scope.qualification_gate(runtime, probe, raw, original)
        self.assertEqual(passed["status"], "PASS"); self.assertEqual(passed["original_allclose_passed"], 47)
        self.assertTrue(all(passed["original_pattern_comparison"]["equal_fields"].values()))
        probe.gate = lambda: dict(status="FAIL", complete=True)
        self.assertFalse(scope.qualification_gate(runtime, probe, raw, original)["performance_eligible"])
        probe.gate = lambda: dict(status="PASS", complete=True)
        values[0]["allfinite"] = False
        self.assertFalse(scope.qualification_gate(runtime, probe, raw, original)["performance_eligible"])
        values[0]["allfinite"] = True; runtime.records = []
        self.assertFalse(scope.qualification_gate(runtime, probe, raw, original)["target_32_rows_reached"])

    def test_scope_identity_and_clean_trace(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); q = td / "qualification"; q.mkdir()
            runtime, engine, runner = fixture(q)
            old_apply = runtime.apply; retained_by_model_hook = runtime
            runtime.enable_validation(); runtime.begin_measurement("qual")
            runtime.validation_results = [dict(layer_name="target", allclose=False)]
            runtime.records = [dict(status="complete", measurement=False, validation_run=True, groups=[], call_id=0)]
            runtime.finalize(); old_trace = (q / "calls.jsonl").read_bytes()
            runtime.unknown_state = 1
            with self.assertRaisesRegex(RuntimeError, "unknown runtime state"):
                scope.reset_artifact_scope(runtime, engine, runner, td / "bad")
            del runtime.unknown_state
            engine.engine_core.engine_core.scheduler.waiting = ["unfinished"]
            with self.assertRaisesRegex(RuntimeError, "drained"):
                scope.reset_artifact_scope(runtime, engine, runner, td / "busy")
            engine.engine_core.engine_core.scheduler.waiting = []
            result = scope.reset_artifact_scope(runtime, engine, runner, td / "performance")
            self.assertTrue(result["allocations_identical"])
            self.assertIs(retained_by_model_hook, runtime); self.assertEqual(old_apply, runtime.apply)
            self.assertEqual(runtime.apply(), (id(runtime), False, 0))
            self.assertFalse(hasattr(runtime, "numerical_localization"))
            self.assertTrue(all("validated" not in e and not e["ever_loaded"] for e in runtime.layers.values()))
            runtime.begin_measurement("formal")
            runtime.records = [dict(status="complete", measurement=True, validation_run=False, groups=[], call_id=0)]
            summary = runtime.finalize()
            self.assertEqual(summary["all_calls"], summary["measurement_calls"])
            self.assertFalse(summary["validation_run"]); self.assertEqual(summary["kernel_validation"], [])
            self.assertEqual((q / "calls.jsonl").read_bytes(), old_trace)

    def test_formal_loop_preserved_and_startup_order(self):
        old = ROOT.parent / "qwen3_static_prefill_v026/source/run_native_pager.py"
        new = ROOT / "source/run_native_pager.py"
        def loop(path):
            main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
            return next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "repeat")
        before, after = loop(old), loop(new)
        # The one added statement only records the already-completed episode count.
        after.body[-1].body.pop()
        self.assertEqual(ast.dump(before), ast.dump(after))
        source = new.read_text(); startup = source[source.index('        # Reproduce qualifier:'):source.index('        for repeat in range(count):')]
        self.assertEqual(startup.count("reset_drained_pager("), 1)
        self.assertLess(startup.index("reset_drained_pager("), startup.index("qualification_capture("))
        self.assertEqual(startup.count("qualification_capture("), 2)
        self.assertLess(startup.index('if not attribution["performance_eligible"]'), startup.index("reset_artifact_scope("))
        self.assertLess(startup.index("pager.finalize()"), startup.index("reset_artifact_scope("))
        original_capture = ROOT.parent / "qwen3_native_qualification_v026/attempt03/source/native_capture.py"
        self.assertEqual(original_capture.read_bytes(), (ROOT / "source/qualification_native_capture.py").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
