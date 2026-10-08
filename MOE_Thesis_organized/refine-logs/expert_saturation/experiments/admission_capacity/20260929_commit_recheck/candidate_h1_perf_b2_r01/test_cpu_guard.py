"""CPU boundary checks for the isolated H1 native-reservation hardening."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent
PKG = ROOT / "pkg"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT.parent))
from c import test_commit_recheck as original_fixture  # noqa: E402


def load_adapter_and_fixture():
    path = PKG / "staged_store_rotation.py"
    spec = importlib.util.spec_from_file_location("h1_guard_qual_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    tree = ast.parse(path.read_text())
    install = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "install")
    start = next(index for index, node in enumerate(install.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "step"
                         for target in node.targets))
    seed = ast.parse(
        "def fixture(scheduler, native, owned, pool, cs, save, commit_recheck):\n"
        " manager=scheduler.kv_cache_manager\n"
        " expected_requests=32\n"
        " open_population=True\n"
        " diagnostic=False\n"
        " store_scope='selected'\n"
        " global_cooldown_steps=20\n"
        " population_mode='open'\n"
        " oldcalc=cs._calc_num_offloadable_tokens\n"
        " hadcalc=True\n"
    ).body[0]
    seed.body += deepcopy(install.body[start:])
    namespace = vars(adapter).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[seed], type_ignores=[])),
                 "<h1-guard-actual-closures>", "exec"), namespace)
    return adapter, namespace["fixture"]


class Harness:
    case = original_fixture.CommitRecheckTest.case
    adapter, fixture = load_adapter_and_fixture()


class ReservationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.harness = Harness()

    def test_zero_preserves_direct_commit_and_native_receipt(self):
        result = self.harness.case(mutate=lambda s, *_:
                                   setattr(s, "_inflight_prefill_reserved_blocks", lambda: 0))
        self.assertEqual(result.data["direct_commits"], 1)
        self.assertEqual(result.data["applied_rotations"], 0)
        self.assertEqual(result.data["native_reservation_gate"],
                         {"checked": 1, "zero": 1, "positive_keep": 0, "unknown_keep": 0})
        self.assertEqual(next(e["native_admission"] for e in result.data["events"]
                              if e["event"] == "direct_commit"), "SCHEDULED_TOKENS")
        result.undo()

    def test_positive_reservation_keeps_original_victim_path(self):
        # In the conditional F=N=3, R=1 boundary, the old direct gate was insufficient.
        result = self.harness.case(free=3, mutate=lambda s, *_:
                                   setattr(s, "_inflight_prefill_reserved_blocks", lambda: 1))
        self.assertEqual(result.data["direct_commits"], 0)
        self.assertEqual(result.data["applied_rotations"], 1)
        self.assertEqual(result.result.preempted_req_ids, {"v"})
        event = next(e for e in result.data["events"] if e["event"] == "commit_recheck")
        self.assertEqual(event["base_reason"], "DIRECT_READY")
        self.assertEqual(event["reason"], "KEEP_NATIVE_INFLIGHT_RESERVATION")
        self.assertEqual(event["native_inflight_reserved_blocks"], 1)
        self.assertEqual(result.data["native_reservation_gate"]["positive_keep"], 1)
        result.undo()

    def test_unknown_reservation_keeps_original_victim_path(self):
        for value in (None, True, -1, "0"):
            with self.subTest(value=value):
                def mutate(s, *_):
                    if value is not None:
                        s._inflight_prefill_reserved_blocks = lambda: value
                result = self.harness.case(mutate=mutate)
                self.assertEqual(result.data["direct_commits"], 0)
                self.assertEqual(result.result.preempted_req_ids, {"v"})
                self.assertEqual(result.data["native_reservation_gate"]["unknown_keep"], 1)
                self.assertEqual(next(e["reason"] for e in result.data["events"]
                                      if e["event"] == "commit_recheck"),
                                 "KEEP_UNKNOWN_NATIVE_RESERVATION")
                result.undo()
        def raises():
            raise RuntimeError("unreadable native reservation")
        result = self.harness.case(mutate=lambda s, *_:
                                   setattr(s, "_inflight_prefill_reserved_blocks", raises))
        self.assertEqual(result.data["direct_commits"], 0)
        self.assertEqual(result.result.preempted_req_ids, {"v"})
        result.undo()

    def test_existing_keep_and_off_do_not_probe_native_reservation(self):
        calls = []
        def mutate(s, *_):
            def probe():
                calls.append(1)
                return 1
            s._inflight_prefill_reserved_blocks = probe
        insufficient = self.harness.case(free=2, mutate=mutate)
        self.assertEqual(calls, [])
        self.assertEqual(insufficient.data["native_reservation_gate"]["checked"], 0)
        self.assertEqual(next(e["reason"] for e in insufficient.data["events"]
                              if e["event"] == "commit_recheck"),
                         "KEEP_INSUFFICIENT_FREE_BLOCKS")
        insufficient.undo()
        off = self.harness.case(enabled=False, mutate=mutate)
        self.assertEqual(calls, [])
        self.assertEqual(off.data["direct_commits"], 0)
        self.assertEqual(off.result.preempted_req_ids, {"v"})
        off.undo()


if __name__ == "__main__":
    subprocess.run([sys.executable, "-B", str(ROOT / "verify_package.py")], check=True)
    subprocess.run(["bash", "-n", str(PKG / "run.sh")], check=True)
    unittest.main()
