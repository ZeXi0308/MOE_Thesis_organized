#!/usr/bin/env python3
"""One stall8/exchange-once ABBA over the frozen whole-group resource controller."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = BASE/'recovery_service_age/run_group.py'
PARENT_SHA = 'c1e30f0eb306b1647e7e2b17302d9c78f9d5f2cca73be43fcb5f8dee8430e29d'


def load_parent():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen service-age group changed')
    spec = importlib.util.spec_from_file_location('exchange_group_parent', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


baseline_action_evidence = load_parent().action_evidence


def action_evidence(data, mode):
    """Count actual donor preemptions; logical release is not GPU-copy completion."""
    if (mode not in ('stall8', 'exchange_once') or data.get('mode') != mode
            or data.get('status') != 'UNINSTALLED' or type(data.get('action_count')) is not int
            or not 0 <= data['action_count'] <= 1 or data.get('max_protected_rounds') != 16
            or not isinstance(data.get('events'), list)):
        raise RuntimeError('Missing or invalid exchange action evidence')
    baseline = baseline_action_evidence(data.get('baseline_stall8', {}), 'stall8')
    events = data['events']; decisions = [e for e in events if e.get('kind') == 'decision']
    actions = [e for e in events if e.get('kind') == 'action']; releases = [e for e in events if e.get('kind') == 'release']
    side_effects = [e for e in events if e.get('kind') in ('action', 'release', 'protection_entry', 'peer_hold', 'target_scheduled')]
    if (len(decisions) > 1 or len(actions) != data['action_count'] or len(releases) != len(actions)
            or (mode == 'stall8' and side_effects)
            or (mode == 'exchange_once' and len(decisions) != len(actions))):
        raise RuntimeError('Shadow, actual preemption, and protection counts disagree')
    if decisions:
        decision = decisions[0]
        if (decision.get('requested_action') is not (mode == 'exchange_once')
                or decision.get('baseline_action') != 'NO_EXCHANGE_FROZEN_STALL8'
                or not isinstance(decision.get('target'), str) or not isinstance(decision.get('donor'), str)
                or decision['target'] == decision['donor']):
            raise RuntimeError('Exchange decision identity or mode differs')
    if actions:
        action, release = actions[0], releases[0]
        donors = [r for r in decision.get('donors', []) if r.get('request') == decision['donor']]
        targets = [r for r in decision.get('targets', []) if r.get('request') == decision['target']]
        if len(donors) != 1 or len(targets) != 1:
            raise RuntimeError('Actual exchange lacks its observed donor/target')
        donor = donors[0]; capacity = decision.get('selected_capacity', {})
        counts = [action.get(k) for k in ('free_before', 'free_after', 'released_blocks', 'donor_num_preemptions')]
        if (action.get('native_preempt_called') is not True or action.get('forced_count') != 1
                or any(type(n) is not int or n < 0 for n in counts) or counts[2] <= 0 or counts[3] < 1
                or counts[1]-counts[0] != counts[2] or counts[2] != donor.get('immediate_releasable_blocks')
                or action.get('pending_store_jobs') != donor.get('pending_store_jobs')
                or capacity != donor.get('capacity') or capacity.get('status') != 'KNOWN'
                or capacity.get('capacity_fit') is not True or capacity.get('margin_blocks', -1) < 0
                or action.get('target') != decision['target'] or action.get('donor') != decision['donor']
                or release.get('target') != decision['target']
                or not events.index(decision) < events.index(action) < events.index(release)):
            raise RuntimeError('Actual native preemption or logical capacity release differs')
        elapsed = release.get('protected_entries')
        if (type(elapsed) is not int or not 0 <= elapsed <= 16
                or release.get('selected_step') != decision.get('step')
                or release.get('step')-decision.get('step') != elapsed
                or any(e.get('target') != decision['target'] or not decision['step'] <= e['step'] <= release['step'] for e in side_effects)):
            raise RuntimeError('Protection is not a bounded window for the selected target')
        if release.get('reason') == 'FIRST_CLIENT_RECEIPT':
            stamp = release.get('target_receipt_host_perf_s')
            if type(stamp) not in (int, float) or not decision['target_receipt_at_selection'] < stamp <= release['host_perf_s']:
                raise RuntimeError('Q1 release lacks a new past client receipt')
    return dict(actual_exchange_count=len(actions), shadow_decision_count=len(decisions)-len(actions),
        target=decisions[0]['target'] if decisions else None, donor=decisions[0]['donor'] if decisions else None,
        release_reasons=[r.get('reason') for r in releases], baseline_stall8=baseline,
        evidence_semantics='Only a recorded successful native donor preemption counts. Free-block change is logical ownership; native STORE flush fences physical reuse. All service costs remain in request metrics.')


def adapted_source():
    parent = load_parent(); text, sources = parent.adapted_source()
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1: raise RuntimeError('Exchange controller boundary changed: '+old[:100])
        text = text.replace(old, new)
    original = inspect.getsource(parent.action_evidence)
    replace(original, original.replace('def action_evidence(', 'def baseline_action_evidence(', 1)+'\n\n'+inspect.getsource(action_evidence))
    replace("['age8', 'stall8', 'stall8', 'age8']", "['stall8', 'exchange_once', 'exchange_once', 'stall8']")
    replace('Requires fixed cap256 age8/stall8/stall8/age8,450s per child,current GPU/110GiB host',
            'Requires fixed cap256 stall8/exchange_once/exchange_once/stall8,450s per child,current GPU/110GiB host')
    replace('B_RECOVERY_SERVICE_AGE=mode,', 'B_RECOVERY_CAPACITY_EXCHANGE=mode,')
    replace("receipt['cells'][-1]['recovery_service_age_mode']", "receipt['cells'][-1]['recovery_capacity_exchange_mode']")
    replace("cell/'output/recovery-service-age.json'", "cell/'output/recovery-capacity-exchange.json'")
    replace("receipt['cells'][-1]['recovery_service_age_actions']", "receipt['cells'][-1]['recovery_capacity_exchange_actions']")
    replace("if i == 1 and evidence['difference_decisions'] == 0:", "if i == 1 and evidence['actual_exchange_count'] == 0:")
    replace('First stall8 had no different executed queue mutation or executable age8 suppression; reverse cells not started; suggestions alone do not count',
            'First exchange_once had no actual native donor preemption; reverse cells not started; shadow trigger does not count')
    return text, [*sources, PARENT, BASE/'recovery_start_gate/progress_capacity.py', Path(__file__)]


def self_check():
    text, _ = adapted_source(); compile(text, '<capacity-exchange-group>', 'exec')
    assert "['stall8', 'exchange_once', 'exchange_once', 'stall8']" in text
    assert 'B_RECOVERY_CAPACITY_EXCHANGE=mode' in text and 'B_RECOVERY_SERVICE_AGE=mode' not in text
    assert "str(ROOT/'run_cell.py')" in text and "cell/'output/recovery-capacity-exchange.json'" in text
    assert "if i == 1 and evidence['actual_exchange_count'] == 0:" in text
    baseline = dict(mode='stall8', status='UNINSTALLED', action_limit=8,
        action_count=0, events=[], bypassed_episodes=[])
    data = dict(mode='stall8', status='UNINSTALLED', action_count=0,
        max_protected_rounds=16, baseline_stall8=baseline, events=[dict(kind='decision',
        target='target', donor='donor', requested_action=False, baseline_action='NO_EXCHANGE_FROZEN_STALL8')])
    evidence = action_evidence(data, 'stall8')
    assert evidence['actual_exchange_count'] == 0 and evidence['shadow_decision_count'] == 1
    data.update(mode='exchange_once', events=[])
    assert action_evidence(data, 'exchange_once')['actual_exchange_count'] == 0
    data['action_count'] = 1
    try: action_evidence(data, 'exchange_once')
    except RuntimeError: pass
    else: raise AssertionError('Actual preemption without action evidence accepted')
    print('PASS: exchange controller ABBA/env/output, actual-preempt continuation, shadow/zero action and missing-action rejection; resource lifecycle unchanged. CPU only.')


def main():
    if sys.argv[1:] == ['--self-check']:
        self_check(); return 0
    parent = load_parent(); parent.adapted_source = adapted_source; parent.__file__ = str(__file__)
    return parent.main()


if __name__ == '__main__': raise SystemExit(main())
