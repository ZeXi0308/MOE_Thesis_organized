#!/usr/bin/env python3
"""BAAB layout metadata around the frozen recovery-fit analyzer; metrics unchanged."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '2aa8681a9ce34a176d24663f28450303c61ad72b5a824507af07a09c0078232c'
EXPECTED = ['fit_once', 'native', 'native', 'fit_once']


def adapt_result(result):
    layout = result['execution_layout']
    layout.pop('expected_abba')
    layout.pop('complete_abba')
    layout.update(expected_baab=EXPECTED.copy(), complete_baab=layout['modes'] == EXPECTED)
    old = 'Two ABBA run contrasts are descriptive;'
    if result['recovery_fit_semantics'].count(old) != 1:
        raise RuntimeError('Frozen analyzer presentation boundary changed')
    result['recovery_fit_semantics'] = result['recovery_fit_semantics'].replace(
        old, 'Two BAAB development-repeat run contrasts are descriptive;')
    result['analyzer_sources_sha256']['recovery_fit/analyze_reverse.py'] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    path = ROOT/'analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-fit analyzer changed')
    spec = importlib.util.spec_from_file_location('recovery_fit_frozen_analysis', path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    original = parent.analyze_session
    parent.analyze_session = lambda session: adapt_result(original(session))
    return parent.main()


if __name__ == '__main__':
    main()
