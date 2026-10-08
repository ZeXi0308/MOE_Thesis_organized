#!/usr/bin/env python3
"""Same bounded native/wait_release group; add identical integer observations."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'd95f2c51003a77781782023275dcb07a4ea159b280d8d6655e5f6b4e35e135dc'


def load_parent():
    path = ROOT/'run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen start-gate group changed')
    spec = importlib.util.spec_from_file_location('observed_start_gate_group', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parent = load_parent()
    original = parent.adapted_source

    def adapted_source():
        text, sources = original()
        old = "str(ROOT/'run_fixed_cell.py'), '--inputs'"
        if text.count(old) != 1:
            raise RuntimeError('Observed group child boundary changed')
        text = text.replace(old, "str(ROOT/'run_observed_fixed_cell.py'), '--inputs'")
        return text, [*sources, Path(__file__)]

    parent.adapted_source = adapted_source
    return parent.main()


if __name__ == '__main__':
    raise SystemExit(main())
