#!/usr/bin/env python3
"""One BAAB reversal of frozen observed start-gate cells and resource controls."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'd1d8325cf4796a066b151bb04b9914bce3e595782a24be4bc38d877dd0e0bb02'
EXPECTED = ['wait_release', 'native', 'native', 'wait_release']


def load_parent():
    path = ROOT/'run_observed_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen observed start-gate controller changed')
    spec = importlib.util.spec_from_file_location('reverse_observed_controller', path)
    observed = importlib.util.module_from_spec(spec); spec.loader.exec_module(observed)
    parent = observed.load_parent(); original = parent.adapted_source

    def adapted_source():
        text, sources = original()
        for old, new in (
            ("p['sequence'] != ['native', 'wait_release', 'wait_release', 'native']",
             "p['sequence'] != "+repr(EXPECTED)),
            ('Requires fixed normal-capacity native/recovery-start-gate ABBA',
             'Requires fixed normal-capacity recovery-start-gate BAAB'),
            ('            if i == 1:\n', '            if i == 0:\n'),
            ('First wait_release had no executed start-gate break; reverse cells not started',
             'First wait_release had no executed start-gate break; remaining BAAB cells not started; no native control executed'),
        ):
            if text.count(old) != 1:
                raise RuntimeError('BAAB controller boundary changed: '+old)
            text = text.replace(old, new)
        return text, [*sources, Path(__file__)]

    parent.adapted_source = adapted_source
    observed.load_parent = lambda: parent
    return observed


def self_check():
    observed = load_parent(); parent = observed.load_parent()
    # Let the original observed main perform its real child-path adaptation;
    # intercept only the final main call, before any resource or process action.
    parent.main = lambda: parent.adapted_source()
    text, _ = observed.main(); tree = ast.parse(text)
    compile(tree, '<reverse-controller-cpu-check>', 'exec')
    assert text.count("str(ROOT/'run_observed_fixed_cell.py'), '--inputs'") == 1
    assert "str(ROOT/'run_fixed_cell.py'), '--inputs'" not in text
    guard = next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                 and "p['sequence'] !=" in ast.unparse(n.test))
    p = dict(sequence=EXPECTED,cap=256,engine_max_num_seqs=256,fixed_gpu_kv_bytes=77242302464)
    condition = compile(ast.Expression(guard.test), '<reverse-sequence-check>', 'eval')
    assert eval(condition, {'p':p}) is False
    assert eval(condition, {'p':p | dict(sequence=['native','wait_release','wait_release','native'])}) is True
    stop = next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='i == 0')
    probe = ast.parse('def probe(counts):\n    receipt={}\n    i=0\n    for unused in [0]:\n        pass\n    return receipt\n')
    probe.body[0].body[2].body=[copy.deepcopy(stop)]
    namespace={};exec(compile(ast.fix_missing_locations(probe),'<reverse-stop-check>','exec'),namespace)
    assert namespace['probe'](dict(action_count=0,executed_breaks=0))['status']=='STOP_NO_ACTION'
    assert namespace['probe'](dict(action_count=1,executed_breaks=10))=={}
    try:namespace['probe'](dict(action_count=None,executed_breaks=0))
    except RuntimeError:pass
    else:raise AssertionError('Missing action evidence accepted')
    resource=parent.load_parent().load_parent().load_parent().load_parent().load_parent()
    assert resource.LOCK_IDENTITY==(2304,4312099778)
    assert resource.MINIMUM_FREE_BYTES==2684354560
    print('PASS: BAAB-only generated controller, identical observed child, first-candidate zero stop/missing abort, frozen common-lock and 2.5GiB resource inheritance. CPU only; no process launched.')


def main():
    if sys.argv[1:]==['--self-check']:
        self_check();return 0
    return load_parent().main()


if __name__=='__main__':
    raise SystemExit(main())
