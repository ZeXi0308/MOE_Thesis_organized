"""Combine both predeclared H1 blocks without selecting repeats or pooling tokens."""
import argparse
import json
from pathlib import Path

from audit_h128_guarded_transfer_r02 import read, require, sha_file
from analyze_h1_performance_pair_r01 import PROTOCOL_SHA, PACKAGE_SHA
from evaluate_goodput import pair


def combine(paths, expected_hashes):
    audits = []
    for block, (path, expected) in enumerate(zip(paths, expected_hashes), 1):
        require(sha_file(path) == expected, f"B{block} audit bytes changed")
        audit = read(path)
        require(audit['status'] == 'H1_PERFORMANCE_PAIR_COMPLETE' and
                audit['block_index'] == block and audit['protocol_sha256'] == PROTOCOL_SHA and
                audit['package_manifest_sha256'] == PACKAGE_SHA,
                f"B{block} not the frozen complete pair")
        for gate in ('off', 'on'):
            m = audit['metrics'][gate]
            require(m['completed'] == m['expected_requests'] == 128 and
                    m['failed'] == m['unfinished'] == 0, 'incomplete full cohort')
        audits.append(audit)
    # Cross-block identity only. This does not make the separate trajectories paired actions.
    pair(audits[0]['metrics']['off'], audits[1]['metrics']['off'])
    blocks = []
    for audit in audits:
        differences = audit['request_comparison']['per_request']
        counts = {}
        for key in ('flow_difference_s', 'ttft_difference_s', 'max_gap_difference_s'):
            values = [r[key] for r in differences if r[key] is not None]
            counts[key] = {'better': sum(v < 0 for v in values),
                           'worse': sum(v > 0 for v in values),
                           'equal': sum(v == 0 for v in values), 'defined': len(values)}
        f = audit['request_comparison']['frontier']
        blocks.append({'block': audit['block_index'],
            'order': ['off','on'] if audit['block_index'] == 1 else ['on','off'],
            'metric_criterion': audit['meets_block_criterion'],
            'output_rate_ratio_on_off': audit['output_rate_ratio_on_off'],
            'mean_flow_ratio_on_off': audit['mean_flow_ratio_on_off'],
            'max_gap_better': audit['max_gap_better'],
            'on_direct_commits': audit['cells']['on']['direct_commits'],
            'off_direct_commits': audit['cells']['off']['direct_commits'],
            'request_differences': counts,
            'frontier_goodput_better': sum(r['goodput_difference_requests_s'] > 0 for r in f),
            'frontier_goodput_worse': sum(r['goodput_difference_requests_s'] < 0 for r in f),
            'output_differences': audit['output_differences']})
    no_actions = all(b['on_direct_commits'] == 0 for b in blocks)
    both_pass = all(b['metric_criterion'] for b in blocks)
    return {'schema_version': 1, 'status': 'TWO_FROZEN_H1_BLOCKS_COMPLETE',
        'audits_sha256': {str(p.name): h for p,h in zip(paths, expected_hashes)},
        'protocol_sha256': PROTOCOL_SHA, 'complete_requests_each': 128,
        'measured_requests_total': 512, 'blocks': blocks,
        'both_blocks_meet_frozen_metric_criterion': both_pass,
        'no_direct_actions_in_either_performance_on_arm': no_actions,
        'interpretation': ('NO_DIRECT_ACTION_IN_FROZEN_PERFORMANCE_DOMAIN: numerical differences cannot establish H1 direct-admission benefit.'
                           if no_actions else
                           'OBSERVED_ACTIONS_WITH_PAIRED_POLICY_RESULTS: assess all block tradeoffs; action presence alone does not establish causal benefit.'),
        'scope': 'Two ordered pairs on previously viewed H128. No token pooling, independent per-action counterfactual, equal-work speedup, quality equivalence, statistical confirmation or blind validation.',
        'next': ('Stop this frozen H1 performance extension without input/seed/threshold tuning to manufacture actions.'
                 if no_actions else 'Interpret the fixed two-block criterion and actual actions before proposing any new experiment.')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1', type=Path, required=True)
    parser.add_argument('--b1-sha256', required=True)
    parser.add_argument('--b2', type=Path, required=True)
    parser.add_argument('--b2-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'new summary output required')
    result = combine([a.b1,a.b2],[a.b1_sha256,a.b2_sha256])
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','both_blocks_meet_frozen_metric_criterion',
                     'no_direct_actions_in_either_performance_on_arm','interpretation')}))
