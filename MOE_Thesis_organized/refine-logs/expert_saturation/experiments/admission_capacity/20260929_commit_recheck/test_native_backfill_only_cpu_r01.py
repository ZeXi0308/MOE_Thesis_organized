#!/usr/bin/env python3
"""Narrow CPU closure checks for the private native-full ordinary candidate."""
from __future__ import annotations

import ast
from pathlib import Path
import runpy
import sys
import textwrap
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent / "candidate_native_backfill_only_r01" / "pkg"
sys.path.insert(0, str(ROOT))
import staged_store_rotation as staged  # noqa: E402


def test_runner_dispatch() -> None:
    runner = runpy.run_path(str(ROOT / "run_recovery_cadence.py"))
    seen = {}

    def installer(_scheduler, **kwargs):
        seen.update(kwargs)
        return {"status": "INSTALLED"}, lambda: None

    result, _ = runner["install_policy"](object(), None, 16,
        "native_full_ordinary_only", False, 0, installer,
        ordinary_backfill=True)
    assert result["status"] == "INSTALLED"
    assert seen["store_scope"] == "native_full"
    assert seen["allow_forced_rotations"] is False
    assert seen["ordinary_backfill"] is True
    assert seen["capacity_victim"] is False

    # The existing native reference returns its no-adapter receipt without
    # invoking staged_store_rotation.install at all.
    scheduler = SimpleNamespace(connector=SimpleNamespace(
        connector_scheduler=SimpleNamespace(config=SimpleNamespace(
            offload_prompt_only=False))))
    def fail_installer(*_args, **_kwargs):
        raise AssertionError("native reference installed the adapter")
    baseline, _ = runner["install_policy"](scheduler, None, 16,
        "native_full_native", False, None, fail_installer)
    assert baseline["store_scope"] == "native_full"
    assert baseline["native_calc_overridden"] is False
    assert baseline["allow_forced_rotations"] is False


def test_full_save_validation_and_selector_guard() -> None:
    # The real installer reaches its later block-size gate, proving that the
    # exact private policy combination passes its early semantic checks.
    try:
        staged.install(None, vllm_config=None, save=True, block_size=0,
            store_scope="native_full", ordinary_backfill=True,
            allow_forced_rotations=False, capacity_victim=False)
    except ValueError as error:
        assert str(error) == "Qualification requires 16-token blocks"
    else:
        raise AssertionError("early validation unexpectedly continued")

    tree = ast.parse((ROOT / "staged_store_rotation.py").read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "install")
    begin = next(n for n in ast.walk(install) if isinstance(n, ast.FunctionDef)
                 and n.name == "begin")
    proposal = next(n for n in ast.walk(begin) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "tracker.decide")
    containing = [n for n in ast.walk(begin) if isinstance(n, ast.If)
                  and proposal in list(ast.walk(n))]
    assert any("allow_forced_rotations" in ast.unparse(n.test)
               for n in containing), "tracker.decide lacks the forced-policy gate"

    calc_writes = [n for n in ast.walk(install) if isinstance(n, ast.Assign)
                   and any("_calc_num_offloadable_tokens" in ast.unparse(t)
                           for t in n.targets)]
    assert len(calc_writes) == 2  # Install-time override and uninstall-time restore.
    for write in calc_writes:
        guards = [n for n in ast.walk(install) if isinstance(n, ast.If)
                  and write in list(ast.walk(n))]
        assert any(ast.unparse(n.test) == "store_scope == 'selected'" for n in guards), (
            "native-full arm could overwrite the full-save class method")


def test_real_ordinary_choice_closure() -> None:
    tree = ast.parse((ROOT / "staged_store_rotation.py").read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "install")
    ordinary = next(n for n in ast.walk(install) if isinstance(n, ast.FunctionDef)
                    and n.name == "attempt_ordinary_backfill")
    source = textwrap.indent(ast.unparse(ordinary), "    ")
    outer = "def exercise(scheduler, pool, cs, tracker, view, manager, owned):\n"
    outer += "    ordinary_choice = None\n"
    outer += "    step = 40\n"
    outer += "    data = {'ordinary_backfill_gate_counts': {}, 'events': [], 'ordinary_backfill_choices': 0}\n"
    outer += "    def start_protection(target, origin):\n"
    outer += "        data['started'] = (target.request_id, origin)\n"
    outer += source + "\n"
    outer += "    selected = attempt_ordinary_backfill()\n"
    outer += "    return selected, ordinary_choice, data\n"
    namespace = {
        "time": staged.time,
        "_native_reservation_disposition": lambda *_: ("DIRECT_READY", 0),
        "_direct_resume_reason": lambda _s, _m, _p, _o, _c, req: (
            "DIRECT_READY" if req.request_id == "candidate" else
            "KEEP_INSUFFICIENT_FREE_BLOCKS"),
    }
    exec(outer, namespace)
    head = SimpleNamespace(request_id="head",
                           status=SimpleNamespace(name="PREEMPTED"),
                           num_prompt_tokens=176, num_output_tokens=0)
    candidate = SimpleNamespace(request_id="candidate",
                                status=SimpleNamespace(name="PREEMPTED"),
                                num_prompt_tokens=80, num_output_tokens=0)
    class Waiting(list):
        def peek_request(self):
            return self[0]
    scheduler = SimpleNamespace(skipped_waiting=[], waiting=Waiting([head, candidate]),
                                running=[])
    pool = SimpleNamespace(get_num_free_blocks=lambda: 10)
    cs = SimpleNamespace(_jobs={}, has_pending_push_work=lambda: False)
    tracker = SimpleNamespace(absent_since={"head": 0, "candidate": 0},
                              config=SimpleNamespace(min_absence_steps=30))
    selected, choice, data = namespace["exercise"](
        scheduler, pool, cs, tracker,
        lambda req: SimpleNamespace(pure_decode=True, remaining_blocks=5),
        object(), {})
    assert selected and choice["target"] == "candidate"
    assert choice["queue_head"] == "head" and choice["free_blocks"] == 10
    assert choice["native_inflight_reserved_blocks"] == 0
    assert data["started"] == ("candidate", "ORDINARY_BACKFILL")
    assert data["ordinary_backfill_choices"] == 1


if __name__ == "__main__":
    test_runner_dispatch()
    test_full_save_validation_and_selector_guard()
    test_real_ordinary_choice_closure()
    print("native-full ordinary CPU closure: 3 checks passed")
