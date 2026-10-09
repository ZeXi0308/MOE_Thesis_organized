"""One maximum-release donor endpoint over the frozen after-failure probe.

Only ranking within its already sufficient/legal donor set changes. The explicit
development target, real failure/next-begin trigger, joint fit, Q1/16-entry bound,
native preempt/STORE/LOAD ownership, and guarded runtime remain inherited.
"""
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import FunctionType

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '3864c586db02b2b895849fd6d25a1d39e31822e3fa85ec2219455ab577e8f96e'
path = ROOT/'exchange_after_failure.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
    raise RuntimeError('Frozen after-failure exchange changed')
spec = importlib.util.spec_from_file_location('max_release_after_failure', path)
PARENT = importlib.util.module_from_spec(spec); spec.loader.exec_module(PARENT)
make_capture, MODES = PARENT.make_capture, PARENT.MODES
RULE = 'Maximum immediate releasable pages; ties: minimum Host missing materialized pages, then native-tail index.'
OLD_SELECTION = "        donor_row = min(sufficient, key=lambda r:(r['release_excess_blocks'],r['host_missing_materialized_blocks'],-r['running_index']))"
NEW_SELECTION = OLD_SELECTION.replace('donor_row =', 'minimum_cost_row =') + "\n        donor_row = min(sufficient, key=lambda r:(-r['immediate_releasable_blocks'],r['host_missing_materialized_blocks'],-r['running_index']))"
OLD_DECISION = "            donor=donor.request_id, selected_capacity=donor_row['capacity'],"
NEW_DECISION = OLD_DECISION + "\n            minimum_cost_donor=minimum_cost_row['request'], maximum_release_donor=donor_row['request'],\n            donor_selection_rule='MAX_RELEASE_HOST_COST_TAIL',"


def _max_source(source):
    for before, after in ((OLD_SELECTION, NEW_SELECTION), (OLD_DECISION, NEW_DECISION)):
        if source.count(before) != 1:
            raise RuntimeError('Frozen donor selection boundary changed')
        source = source.replace(before, after)
    return source


def _attach_for_trigger(trigger):
    source = inspect.getsource(PARENT._attach_for_trigger)
    before = '    source = inspect.getsource(FROZEN._attach)'
    if source.count(before) != 1:
        raise RuntimeError('Frozen failure builder changed')
    source = source.replace(before, before+'\n    source = _max_source(source)')
    namespace = dict(PARENT.__dict__, _max_source=_max_source)
    exec(compile(source, str(__file__)+'[frozen-failure-builder]', 'exec'), namespace)
    return namespace['_attach_for_trigger'](trigger)


def install(scheduler, mode='stall8', *, last_receipts, selective, allocation_observation):
    inherited = FunctionType(PARENT.install.__code__,
        dict(PARENT.__dict__, _attach_for_trigger=_attach_for_trigger),
        'install', PARENT.install.__defaults__)
    inherited.__kwdefaults__ = PARENT.install.__kwdefaults__
    data, uninstall = inherited(scheduler, mode, last_receipts=last_receipts,
        selective=selective, allocation_observation=allocation_observation)
    data['max_release_adapter'] = dict(
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_after_failure_sha256=PARENT_SHA, rule=RULE,
        scope='Same sufficient/legal donor set and joint accounting; one ranking change only. Stall8 records the same exchange shadow without exchanging.')
    data['donor_cost_definition'] = data['donor_cost_definition'].replace(
        'Rank release excess, missing materialized whole pages, native-tail tie.',
        'Rank maximum immediate release, missing materialized whole pages, native-tail tie; original minimum-cost suggestion is retained in each decision.')
    return data, uninstall
