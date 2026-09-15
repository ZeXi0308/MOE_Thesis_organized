#!/usr/bin/env python3
"""Keep synthetic action qualification separate from one real prestate certificate."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'experiments/admission_capacity'))
from service_window_model import Action, Node, Request, State, qualify, retention_efficiency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('output exists; preserve previous result')
    s = State(Request('target', 32, 16, 1, 3),
              (Request('peer', 31, 30, 2, 4),), 16, 4, 100, 120)
    a = Action('mixed-4', 4, (Node('queue', 1), Node('restore', 5, ('queue',))),
               1, ('peer',), (('peer', (2, 4)),), recompute_tokens=16,
               prospective_tax_ms=4)
    def q(state, action):
        return qualify(state, action, gap_budget_ms=10, tax_budget_ms_per_output=1)
    support = [q(s, replace(a, name='mixed-1', tokens=1)), q(s, a),
               q(s, replace(a, name='solo-4', co_batch=(), recovery_peer_outputs=()))]
    low = replace(s, free_gpu_blocks=2)
    counter = [q(low, replace(a, name=f'mixed-{n}', tokens=n, recovery_peer_outputs=()))
               for n in (1, 2, 3, 4, 8)]
    source = args.source_root / ('refine-logs/expert_saturation/outputs/admission_capacity/'
        '20260914_restore_completion_r01/analysis/localization/first_dispatch.json')
    observed = json.loads(source.read_text())
    assert observed['status'] == 'STRUCTURAL_ONE_STEP_DIAGNOSTIC' and observed['step'] == 406
    certificate = []
    for c in observed['cells']:
        need = c['target_remaining_history_blocks'] + c['ready_growth_blocks']
        slack = c['free_blocks'] - need
        assert slack == c['diagnostic_reservation_slack'] == 3
        assert sum(c['diagnostic_tokens'].values()) == 1024 and not c['diagnostic_victims']
        certificate.append(dict(label=c['label'], raw_sha256=c['raw_sha256'],
            free_blocks=c['free_blocks'], recovery_reservation=c['target_remaining_history_blocks'],
            peer_growth=c['ready_growth_blocks'], slack=slack, token_budget=1024,
            scope='ONE_ACTION_ONLY; future service window, tax and timing not established'))
    result = dict(status='STRUCTURAL_MODEL_ONLY', synthetic_state=asdict(s),
        synthetic_action=asdict(a), synthetic_budgets=dict(gap_ms=10, tax_ms_per_output=1),
        support=support, counterexample=dict(state=asdict(low), actions=counter),
        retention_examples=[retention_efficiency(3, 1, p, (10, 10))
                            for p in ((0, 0), (1, 1), (0, 1))],
        observed_source=str(source), observed_step406=certificate,
        boundary='Synthetic milliseconds are not measurements; no actual EOS or future trace is used.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], support=[(x['action'], x['eligible'])
          for x in support], counterexample_eligible=sum(x['eligible'] for x in counter),
          observed_certificates=len(certificate))))


if __name__ == '__main__':
    main()
