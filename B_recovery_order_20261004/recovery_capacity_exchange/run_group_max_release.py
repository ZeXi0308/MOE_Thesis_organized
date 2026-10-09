#!/usr/bin/env python3
"""Change donor selection only; retain the proven after-failure group controller."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
path = ROOT/'run_group_after_failure.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != 'de3336dc762ceb2972fa8fed0b7ad35ba70df8f5b51a17c7ada62191a5c0f946':
    raise RuntimeError('Frozen after-failure controller changed')
spec = importlib.util.spec_from_file_location('max_release_parent_group', path)
PARENT = importlib.util.module_from_spec(spec); spec.loader.exec_module(PARENT)
PARENT_SOURCE = PARENT.adapted_source


def adapted_source():
    text, sources = PARENT_SOURCE()
    before = "str(ROOT/'run_cell_after_failure.py')"
    if text.count(before) != 1: raise RuntimeError('Frozen child entry changed')
    text = text.replace(before, "str(ROOT/'run_cell_max_release.py')")
    before = "    evidence['controlled_failure_trigger']=trigger"
    extra = '''    if decisions:
        eligible = [r for r in d['donors'] if r['capacity']['capacity_fit']]
        maximum = min(eligible, key=lambda r: (-r['immediate_releasable_blocks'], r['host_missing_materialized_blocks'], -r['running_index']))
        minimum = min(eligible, key=lambda r: (r['release_excess_blocks'], r['host_missing_materialized_blocks'], -r['running_index']))
        if (d.get('maximum_release_donor') != maximum['request']
                or d.get('minimum_cost_donor') != minimum['request']
                or d.get('donor') != maximum['request']):
            raise RuntimeError('Executed/shadow donor is not the fresh maximum-release endpoint')
        evidence['minimum_cost_donor'] = minimum['request']
        evidence['maximum_release_donor'] = maximum['request']
        evidence['donor_selection_changed'] = minimum['request'] != maximum['request']
'''
    if text.count(before) != 1: raise RuntimeError('Frozen action evidence boundary changed')
    text = text.replace(before, extra+before)
    return text, [*sources, ROOT/'run_group_after_failure.py', ROOT/'run_cell_max_release.py',
                  ROOT/'exchange_max_release.py', ROOT/'check_max_release_cpu.py', Path(__file__)]


def main():
    if sys.argv[1:] == ['--self-check']:
        text, _ = adapted_source(); compile(text, '<max-release-controller>', 'exec')
        assert "str(ROOT/'run_cell_max_release.py')" in text
        assert "if i == 1 and evidence['actual_exchange_count'] == 0:" in text
        assert "child.wait(timeout=p['per_cell_seconds'])" in text
        print('PASS: maximum-release action check, original bounded group and zero-action stop; CPU only.')
        return 0
    PARENT.adapted_source = adapted_source; PARENT.__file__ = str(__file__)
    return PARENT.main()


if __name__ == '__main__': raise SystemExit(main())
