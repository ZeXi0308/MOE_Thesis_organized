"""CPU checks for this signal ablation; never imports a serving/GPU package."""
import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_TRACES = (
    "high-normal-r01/03_feedback/raw.json",
    "high-normal-r01/04_feedback/raw.json",
    "extra-controls-high/05_feedback/raw.json",
    "extra-controls-high/06_feedback/raw.json",
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definition(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == name),
                    include_attributes=False)


def main():
    original = load("original_confirmation_policy", ROOT.parent / "prefill_policy.py")
    candidate = load("elapsed_ablation_policy", ROOT / "prefill_policy.py")
    assert definition(ROOT / "prefill_policy.py", "install") == definition(
        ROOT.parent / "prefill_policy.py", "install")
    assert definition(ROOT / "run_confirmation.py", "episode") == definition(
        ROOT.parent / "run_confirmation.py", "episode")
    events = 0
    for name in OLD_TRACES:
        raw = json.loads((ROOT.parents[1] / name).read_text())
        a, b = original.Policy("feedback", 24), candidate.Policy("feedback", 24)
        for step in raw["steps"]:
            ndecode = step["decode_count_before"]
            assert a.choose(ndecode) == b.choose(ndecode) == step["budget"]
            elapsed = step["end_s"] - step["start_s"]
            args = (elapsed, step["prefill_tokens"], step["decode_tokens"])
            a.observe(*args)
            b.observe(*args)
            assert (a.index, a.recent_ms, a.levels) == (b.index, b.recent_ms, b.levels)
            events += 1

    # Explicit boundary expectations, including both sides of every switch.
    boundaries = {0: 1024, 1: 2048, 8: 2048, 9: 1024, 72: 1024,
                  73: 512, 112: 512, 113: 256, 120: 256}
    replay = candidate.Policy("elapsed_replay", 24)
    # An object that cannot be multiplied also proves R never consumes elapsed.
    ignored_elapsed = object()
    for mixed in range(121):
        assert replay.mixed_steps == mixed
        assert replay.choose(0) == 2048
        assert replay.choose(1) == replay.choose(64)
        if mixed in boundaries:
            assert replay.choose(1) == boundaries[mixed]
        for prefill, decode in ((0, 64), (2048, 0), (0, 0)):
            before = replay.choose(1)
            replay.observe(ignored_elapsed, prefill, decode)
            assert replay.mixed_steps == mixed and replay.choose(1) == before
        replay.observe(ignored_elapsed, 256, 32)
        assert replay.mixed_steps == mixed + 1
    assert replay.recent_ms is None and replay.index == 2
    print(json.dumps(dict(status="PASS", original_A_event_checks=events,
        old_traces=len(OLD_TRACES), R_boundary_checks=len(boundaries),
        R_nonmixed_observe_checks=121 * 3, R_mixed_advances=121,
        scheduler_install_unchanged=True, episode_unchanged=True)))


if __name__ == "__main__":
    main()
