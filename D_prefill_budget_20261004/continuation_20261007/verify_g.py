"""Targeted CPU checks for the single minimum-level recovery condition."""
import importlib.util
import random
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


root = Path(__file__).resolve().parent
old = module(root.parent / 'confirmation_v2/ablation_elapsed_v3/prefill_policy.py', 'old_policy')
new = module(root / 'prefill_policy.py', 'new_policy')
rng = random.Random(20261007)
for name in ('fixed384', 'feedback', 'elapsed_replay'):
    a, b = old.Policy(name, 24), new.Policy(name, 24)
    for _ in range(1000):
        d = rng.choice([0, 1, 24, 35, 64])
        assert a.choose(d) == b.choose(d)
        elapsed, p = rng.uniform(.008, .070), rng.choice([0, 128, 256, 1024])
        a.observe(elapsed, p, d)
        b.observe(elapsed, p, d)
        assert vars(a) == vars(b)

cases = [
    (0, .020, 256, 35, 0, 1),
    (0, .024, 256, 35, 0, 0),
    (0, .025, 256, 35, 0, 0),
    (0, .016, 256, 35, 1, 1),
    (1, .020, 512, 35, 1, 1),
    (2, .025, 1024, 35, 1, 1),
    (0, .020, 0, 35, 0, 0),
    (0, .020, 256, 0, 0, 0),
]
for index, elapsed, p, d, expected_a, expected_g in cases:
    a, g = new.Policy('feedback', 24), new.Policy('feedback_floor_recover', 24)
    a.index = g.index = index
    a.observe(elapsed, p, d)
    g.observe(elapsed, p, d)
    assert (a.index, g.index) == (expected_a, expected_g)
    assert g.choose(0) == a.choose(0) == 2048
assert (root / 'run_confirmation.py').read_bytes() == (
    root.parent / 'confirmation_v2/ablation_elapsed_v3/run_confirmation.py').read_bytes()
print('PASS: 3000 unchanged baseline transitions; 8 G boundary/mixed-sample cases; identical serving driver.')
