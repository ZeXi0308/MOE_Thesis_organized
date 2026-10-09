#!/usr/bin/env python3
"""Bounded repeat8/unique8 comparison over the frozen host/resource controller."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = BASE/'recovery_start_gate/run_group.py'
PARENT_SHA = 'd95f2c51003a77781782023275dcb07a4ea159b280d8d6655e5f6b4e35e135dc'


def load_parent():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen host/resource controller changed')
    spec = importlib.util.spec_from_file_location('repeat_unique_group_parent', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def action_evidence(data, mode):
    """Validate recorded mutations; distinguish suggestions from legal suppression.

    Suppression eligibility is evaluated at that event in the observed run, not
    by stitching a hypothetical repeat8 trajectory through later events.
    """
    count = data.get('action_count')
    if (data.get('mode') != mode or data.get('status') != 'UNINSTALLED'
            or type(count) is not int or not 0 <= count <= 8
            or data.get('action_limit') != 8 or not isinstance(data.get('events'), list)):
        raise RuntimeError('Missing or invalid repeat-unique action evidence')
    episodes, ids = [], set()
    suggested = different_moves = suppressions = 0
    for index, event in enumerate(data['events']):
        before, after = event.get('waiting_before'), event.get('waiting_after')
        if (event.get('kind') != 'fit_opportunity' or not isinstance(before, list)
                or not before or not all(isinstance(rid, str) for rid in before)
                or len(set(before)) != len(before) or not isinstance(after, list)
                or sorted(before) != sorted(after)
                or event.get('action_count_before') != len(episodes)):
            raise RuntimeError('Invalid repeat-unique queue/count evidence at event '+str(index))
        repeat, unique = event.get('repeat8_suggestion'), event.get('unique8_suggestion')
        rows = event.get('candidates')
        if not isinstance(rows, list):
            raise RuntimeError('Missing repeat-unique legal-set observation')
        eligible = {row['request']: row for row in rows if row.get('eligible') is True}
        if repeat not in eligible or type(eligible[repeat].get('num_preemptions')) is not int:
            raise RuntimeError('Repeat suggestion lacks legal episode evidence')
        repeat_episode = (repeat, eligible[repeat]['num_preemptions'])
        repeat_executable = len(episodes) < 8 and repeat_episode not in episodes
        if (event.get('repeat8_suggestion_num_preemptions') != repeat_episode[1]
                or event.get('repeat8_would_execute') is not repeat_executable):
            raise RuntimeError('Repeat executable suggestion disagrees with actual prior actions')
        suggested += repeat != unique
        reasons = event.get('action_skip_reasons')
        if event.get('action') == 'REORDERED':
            chosen = event.get('candidate_head')
            if chosen not in eligible:
                raise RuntimeError('Executed candidate is not in the observed legal set')
            episode = (chosen, eligible[chosen].get('num_preemptions'))
            if (type(episode[1]) is not int or episode in episodes or len(episodes) >= 8
                    or chosen != (repeat if mode == 'repeat8' else unique)
                    or chosen == before[0] or after != [chosen]+[rid for rid in before if rid != chosen]
                    or event.get('final_head') != chosen or event.get('queue_changed') is not True
                    or reasons != [] or event.get('action_count_after') != len(episodes)+1
                    or (mode == 'unique8' and chosen in ids)):
                raise RuntimeError('Recorded mutation violates the fixed repeat-unique contract')
            episodes.append(episode); ids.add(chosen)
            different_moves += chosen != repeat
        elif event.get('action') == 'SHADOW_ONLY':
            if (after != before or event.get('queue_changed') is not False
                    or event.get('final_head') != before[0]
                    or event.get('action_count_after') != len(episodes)):
                raise RuntimeError('Shadow event reports a queue mutation or consumed budget')
            if (mode == 'unique8' and reasons == ['NO_UNUSED_REQUEST_IN_LEGAL_SET']
                    and unique is None and event.get('candidate_head') is None
                    and set(eligible).issubset(ids) and repeat_executable):
                suppressions += 1
        else:
            raise RuntimeError('Incomplete or failed repeat-unique action event')
    used = data.get('used_request_ids')
    expected_episodes = [dict(request=rid, num_preemptions=number) for rid, number in episodes]
    if (count != len(episodes) or used != sorted(ids)
            or data.get('bypassed_episodes') != expected_episodes):
        raise RuntimeError('Top-level repeat-unique actions disagree with event evidence')
    return dict(reorder_count=count, unique_ids=sorted(ids), unique_id_count=len(ids),
        suggestion_difference_count=suggested,
        different_queue_mutations=different_moves,
        executable_repeat_suppressions=suppressions,
        difference_decisions=different_moves+suppressions,
        evidence_semantics='Recorded queue mutations and same-observed-state executable repeat8 suppressions are distinct; suggestions alone are not interventions or measured savings.')


def adapted_source():
    text, sources = load_parent().adapted_source()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Repeat-unique controller boundary changed: '+old[:100])
        text = text.replace(old, new)

    replace("p['sequence'] != ['native', 'wait_release', 'wait_release', 'native'] or p['cap'] != 256",
            "p['sequence'] != ['repeat8', 'unique8', 'unique8', 'repeat8'] or p['cap'] != 256 or p['per_cell_seconds'] != 450 or p['host_limit_bytes'] != 118111600640 or p['gpu_uuid'] != 'GPU-51b8e4bb-27b8-4b82-5254-7317aae7298c'")
    replace('Requires fixed normal-capacity native/recovery-start-gate ABBA',
            'Requires fixed cap256 repeat8/unique8/unique8/repeat8,450s per child,current GPU/110GiB host')
    replace("str(ROOT/'run_fixed_cell.py'), '--inputs'", "str(ROOT/'run_cell.py'), '--inputs'")
    replace('B_RECOVERY_START_GATE=mode,', "B_RECOVERY_REPEAT_UNIQUE=mode, B_RECOVERY_REPEAT='once',")
    replace("parser.add_argument('--wait-lock-seconds', type=float, default=0)",
            "parser.add_argument('--wait-lock-seconds', type=float, default=1800)")
    replace('    args = parser.parse_args()',
            "    args = parser.parse_args()\n    if args.wait_lock_seconds != 1800:\n        parser.error('This bounded comparison requires wait-lock-seconds1800')")
    replace('def main():', inspect.getsource(action_evidence)+'\n\ndef main():')
    old = """            receipt['cells'][-1]['recovery_start_gate_mode'] = mode
            receipt['cells'][-1]['output_event_storage_mode'] = 'compact'
            action = json.loads((cell/'output/recovery-start-gate.json').read_text())
            counts = {key: action.get(key) for key in ('action_count', 'executed_breaks')}
            receipt['cells'][-1]['recovery_start_gate_action_counts'] = counts
            if i == 1:
                if any(type(value) is not int for value in counts.values()):
                    raise RuntimeError('First wait_release action evidence missing; no remainder started')
                if counts['action_count'] == 0 or counts['executed_breaks'] == 0:
                    receipt.update(status='STOP_NO_ACTION', stop_reason='First wait_release had no executed start-gate break; reverse cells not started')
                    break"""
    new = """            receipt['cells'][-1]['recovery_repeat_unique_mode'] = mode
            receipt['cells'][-1]['output_event_storage_mode'] = 'compact'
            action = json.loads((cell/'output/recovery-repeat-unique.json').read_text())
            evidence = action_evidence(action, mode)
            receipt['cells'][-1]['recovery_repeat_unique_actions'] = evidence
            write(session/'receipt.json', receipt)
            if i == 1 and evidence['difference_decisions'] == 0:
                receipt.update(status='STOP_NO_ACTION', stop_reason='First unique8 had no different executed queue mutation or executable repeat8 suppression; reverse cells not started; suggestions alone do not count')
                break"""
    replace(old, new)
    return text, [*sources, PARENT, BASE/'recovery_repeat/repeat_fit.py', Path(__file__)]


def self_check():
    """Run the whole assembled controller with synthetic children and resources."""
    import contextlib
    import io
    import json
    import os
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    parent = load_parent()
    resource = parent.load_parent().load_parent().load_parent().load_parent().load_parent()
    assert resource.LOCK_IDENTITY == (2304, 4312099778)
    assert resource.LOCK_PATH == '/root/autodl-tmp/moe-research-gpu.lock'
    assert resource.MINIMUM_FREE_BYTES == 2684354560
    text, sources = adapted_source()

    def fixture(mode, scenario):
        data = dict(mode=mode, status='UNINSTALLED', action_limit=8, action_count=0,
                    used_request_ids=[], bypassed_episodes=[], events=[])
        if scenario == 'zero':
            return data
        def event(chosen, repeat, number=1, reasons=None):
            count = data['action_count']; before = ['head', 'r1', 'r2']
            rows = [dict(request=rid, eligible=bool(chosen) or rid=='r1', num_preemptions=number) for rid in ('r1','r2')]
            after = [chosen]+[rid for rid in before if rid != chosen] if chosen else before[:]
            e = dict(kind='fit_opportunity', waiting_before=before, waiting_after=after,
                     repeat8_suggestion=repeat, unique8_suggestion=chosen,
                     repeat8_suggestion_num_preemptions=number,
                     repeat8_would_execute=count<8 and dict(request=repeat,num_preemptions=number) not in data['bypassed_episodes'],
                     candidate_head=chosen, candidates=rows, final_head=after[0],
                     queue_changed=chosen is not None, action='REORDERED' if chosen else 'SHADOW_ONLY',
                     action_skip_reasons=[] if chosen else reasons,
                     action_count_before=count, action_count_after=count+(chosen is not None))
            data['events'].append(e)
            if chosen:
                data['action_count'] += 1
                data['used_request_ids'] = sorted(set(data['used_request_ids']) | {chosen})
                data['bypassed_episodes'].append(dict(request=chosen, num_preemptions=number))
        event('r1', 'r1')
        if mode == 'unique8':
            if scenario == 'move': event('r2', 'r1', 2)
            elif scenario in ('suppress','already_used_episode'):
                event(None, 'r1', 2 if scenario == 'suppress' else 1, ['NO_UNUSED_REQUEST_IN_LEGAL_SET'])
            elif scenario == 'invalid': data['action_count'] = 8
        return data

    for scenario, expected in [('move','COMPLETE'), ('suppress','COMPLETE'),
                               ('zero','STOP_NO_ACTION'), ('already_used_episode','STOP_NO_ACTION'),
                               ('invalid','ABORTED')]:
        with tempfile.TemporaryDirectory(prefix='b-repeat-unique-controller-') as tmp:
            tmp = Path(tmp); session = tmp/'session'; overlay = tmp/'overlay'; overlay.mkdir()
            (overlay/'repair.json').write_text(json.dumps(dict(status='CPU_IMPORT_OK')))
            plan = dict(session_dir=str(session), sequence=['repeat8','unique8','unique8','repeat8'],
                        cap=256, engine_max_num_seqs=256, fixed_gpu_kv_bytes=77242302464,
                        per_cell_seconds=450, host_limit_bytes=118111600640,
                        gpu_uuid='GPU-51b8e4bb-27b8-4b82-5254-7317aae7298c',
                        lock_path=resource.LOCK_PATH, hf_cache=str(tmp/'hf'), library_path='',
                        runtime_overlay=str(overlay), python='synthetic-child',
                        inputs_dir=str(tmp/'inputs'), measurement_max_seconds=360)
            plan_path = tmp/'plan.json'; plan_path.write_text(json.dumps(plan))
            ns = dict(__name__='synthetic_repeat_unique_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
            exec(compile(text, str(__file__)+'[self-check]', 'exec'), ns)
            launched, acquired, closed, boundaries = [], [], [], []
            def acquire(fd, seconds, receipt, path):
                acquired.append((fd, seconds)); receipt['status'] = 'RUNNING'
            def boundary(uuid):
                boundaries.append(uuid); return dict(empty=True, synthetic=True)
            def popen(command, **kwargs):
                assert acquired == [(99,1800)] and kwargs['pass_fds'] == (9,)
                assert command[1] == str(ROOT/'run_cell.py')
                env = kwargs['env']; mode = env['B_RECOVERY_REPEAT_UNIQUE']
                assert env['B_RECOVERY_REPEAT'] == 'once' and env['B_OUTPUT_EVENT_STORAGE'] == 'compact'
                assert 'B_RECOVERY_START_GATE' not in env
                output = Path(command[command.index('--output-dir')+1]); output.mkdir()
                (output/'memory-after-init.json').write_text(json.dumps(dict(kv_storage_bytes=77242302464)))
                (output/'recovery-repeat-unique.json').write_text(json.dumps(fixture(mode,scenario)))
                launched.append(mode)
                def wait(timeout):
                    assert timeout == 450; return 0
                return SimpleNamespace(wait=wait, pid=123)
            fake_os = SimpleNamespace(**vars(os))
            fake_os.open = lambda *a, **k: 99
            fake_os.fstat = lambda fd: SimpleNamespace(st_dev=2304, st_ino=4312099778)
            fake_os.dup2 = lambda *a, **k: None
            fake_os.close = closed.append
            ns.update(acquire_gpu_lock=acquire, os=fake_os,
                      subprocess=SimpleNamespace(Popen=popen, STDOUT=-2),
                      base=SimpleNamespace(checked_boundary=boundary))
            ns['Path'] = lambda path: (SimpleNamespace(read_text=lambda:'118111600640')
                                      if str(path)=='/sys/fs/cgroup/memory.max' else Path(path))
            with patch.object(sys, 'argv', [str(__file__),'--plan',str(plan_path)]), contextlib.redirect_stdout(io.StringIO()):
                result = ns['main']()
            receipt = json.loads((session/'receipt.json').read_text())
            assert receipt['status'] == expected, receipt
            assert result == (1 if expected == 'ABORTED' else 0)
            assert launched == plan['sequence'][:4 if expected == 'COMPLETE' else 2]
            assert acquired == [(99,1800)] and closed == [9,99]
            assert len(boundaries) == len(launched)+(expected != 'ABORTED')
            if scenario == 'suppress':
                e = receipt['cells'][1]['recovery_repeat_unique_actions']
                assert e['different_queue_mutations'] == 0 and e['executable_repeat_suppressions'] == 1
            if scenario == 'already_used_episode':
                e = receipt['cells'][1]['recovery_repeat_unique_actions']
                assert e['suggestion_difference_count'] == 1 and e['difference_decisions'] == 0
    print('PASS: complete synthetic ABBA, actual different move, executable suppression, zero-action/only-suggestion stop, malformed evidence abort,450s children/1800s lock,final release; frozen common lock/2.5GiB/current GPU/110GiB. CPU only; no GPU command executed.')


def main():
    if sys.argv[1:] == ['--self-check']:
        self_check(); return 0
    parent = load_parent()
    parent.adapted_source = adapted_source
    parent.__file__ = str(__file__)
    return parent.main()


if __name__ == '__main__':
    raise SystemExit(main())
