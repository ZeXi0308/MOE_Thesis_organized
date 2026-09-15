"""Price mandatory peer withholding in an existing conditional bridge branch.

Reads retained branch results and their native prestate; no new trajectory,
primary-service reanalysis, target search, future EOS, or wall-clock estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path

from recovery_execution_share import Request, State, decode_service_envelope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base = args.repo/'refine-logs/expert_saturation/outputs/admission_capacity'
    old = base/'20260915_recovery_execution_share_r01/native_full_release_boundary/analysis.json'
    rawpath = base/'20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full/raw.json'
    prior, raw = json.loads(old.read_text()), json.loads(rawpath.read_text())
    aliases = raw['internal_to_source']
    request = {r['request_id']: r for r in raw['requests']}
    result = []
    for row in prior['states']:
        branch = row['cap_bridge']
        if branch['status'] != 'CAP_OUTPUT_BOUNDARY_NOT_PHYSICAL_RELEASE':
            continue
        before = raw['memory_trace'][row['step']]['before']
        def view(rid):
            r = before['requests'][rid]
            return Request(aliases[rid], r['prompt_tokens']+r['output_tokens'],
                r['computed_tokens'], sum(r['block_counts']), r['output_tokens'],
                request[aliases[rid]]['max_output_tokens']-r['output_tokens'])
        rows = [view(rid) for rid in before['running_ids']]
        target = next(r for r in rows if r.request_id == row['target'])
        state = State(target, tuple(r for r in rows if r is not target), before['pool']['free_blocks'])
        required = {target.request_id, *branch['completed_at_bound']}
        bound = decode_service_envelope(state, len(branch['calls']), required)
        outputs = branch['new_output_opportunities']
        assert sum(outputs.values()) == bound['maximum_total_output_opportunities']
        zero = [rid for rid, n in outputs.items() if n == 0]
        assert len(zero) == bound['minimum_zero_service_peers']
        result.append(dict(step=row['step'], bound=bound,
            achieved_total_output_opportunities=sum(outputs.values()),
            observed_in_cpu_branch_zero_service_peers=zero,
            per_request_output_opportunities=outputs))
    paths = [old, rawpath, Path(__file__), Path(__file__).with_name('recovery_execution_share.py')]
    payload = dict(status='CONDITIONAL_PEER_COST_BOUND', states=result,
        sources={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
        limitations='Same retained diagnostic branch, not an independent experiment. '
        'No early EOS, held-block return, arrival/admission, preemption or transfer changes. '
        'This bounds output opportunities and call withholding, not time, quality or complete-service utility.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(payload, f, indent=2); f.write('\n')
    print(json.dumps([dict(step=x['step'], total=x['achieved_total_output_opportunities'],
        unavoidable_zero_peers=x['bound']['minimum_zero_service_peers'],
        minimum_withheld=x['bound']['minimum_total_withheld_opportunities']) for x in result]))


if __name__ == '__main__':
    main()
