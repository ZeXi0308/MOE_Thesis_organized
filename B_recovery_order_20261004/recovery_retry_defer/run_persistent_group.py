#!/usr/bin/env python3
"""Change only the child entrypoint of the frozen 3GiB retry-defer controller."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '39bfe256430e3af2d2681ed99e8560f732942aade7cb00f7e0d97d06bc01bab6'


def load_parent():
    path = ROOT/'run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen retry-defer controller changed')
    spec = importlib.util.spec_from_file_location('persistent_retry_group_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text, sources):
    old = "str(ROOT/'run_fixed_cell.py'), '--inputs'"
    if text.count(old) != 1:
        raise RuntimeError('Persistent retry controller child boundary changed')
    return text.replace(old, "str(ROOT/'run_persistent_fixed_cell.py'), '--inputs'"), sources


def adapted_source():
    return adapt_source(*load_parent().adapted_source())


def main():
    parent = load_parent(); original = parent.adapted_source
    parent.adapted_source = lambda: adapt_source(*original())
    return parent.main()


if __name__ == '__main__':
    raise SystemExit(main())
