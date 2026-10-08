"""Posthoc criterion replay on observed PF states; no policy counterfactual."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from C_NATIVE_RETIREMENT_ADMISSION_V1 import decode_envelope
from C_PAST_FUTURE_AE_PREDICTOR_V1 import AERequest, normal_req_tuple, future_peak_tokens


def analyze(cell):
    source = cell / 'past-future-ae.json'
    data = source.read_bytes()
    trace = json.loads(data)
    assert trace['status'] == 'DRAINED' and not trace['violations']
    actual = {event['call']: event for event in trace['events']
              if event['event'] in ('admit', 'resume')}
    rows = []
    for decision in trace['decisions']:
        residents = decision['running_inputs']
        head = decision['head_input']
        converted = [SimpleNamespace(num_prompt_tokens=r['prompt_tokens'],
            num_output_tokens=r['output_tokens'], max_tokens=r['max_output_tokens'])
            for r in residents]
        envelope = decode_envelope(converted)
        head_cost = (head['prompt_tokens'] + head['max_output_tokens'] + 15) // 16
        feasible = len(residents) < 32 and decision['joint_batch_need'] <= 4096
        cap_tuples = [normal_req_tuple(AERequest(**r), r['max_output_tokens'])
                      for r in residents + [head]]
        full_cap_peak = future_peak_tokens(cap_tuples)
        allowed = decision['allowed']
        full_cap_ae = feasible and full_cap_peak < decision['token_limit']
        legacy = feasible and envelope + head_cost <= 4096
        event = actual.get(decision['call'])
        if event is not None:
            assert allowed and event['request_id'] == decision['head']
        rows.append(dict(call=decision['call'], head=decision['head'],
            resident_count=len(residents), reason=decision['reason'],
            compatible=feasible, pf_allowed=allowed,
            actual_admission=event is not None,
            legacy_cap_feasible=legacy, resident_envelope_blocks=envelope,
            legacy_total_blocks=envelope + head_cost,
            ae_full_cap_allowed=full_cap_ae, ae_full_cap_peak_tokens=full_cap_peak,
            sampled_peak_tokens=decision['sampled_peak'], token_limit=decision['token_limit'],
            history_shorter_than_cap=sum(n < 1024 for n in decision['history_before'])))
    def table(flag):
        result = {}
        for pf in (False, True):
            for alternate in (False, True):
                selected = [r for r in rows if r['pf_allowed'] == pf and r[flag] == alternate]
                result[f'pf_{int(pf)}_alternate_{int(alternate)}'] = dict(
                    calls=len(selected), unique_heads=len({r['head'] for r in selected}),
                    actual_admissions=sum(r['actual_admission'] for r in selected),
                    first_example=selected[0] if selected else None)
        return result
    assert len(actual) == sum(r['actual_admission'] for r in rows) == 128
    return dict(schema='c-native-past-future-ae-state-compare-v1',
        status='POSTHOC_CRITERION_REPLAY_ON_OBSERVED_PF_TRAJECTORY',
        source_sha256=hashlib.sha256(data).hexdigest(),
        decision_calls=len(rows), reasons=dict(Counter(r['reason'] for r in rows)),
        resident_cap_envelope_over_capacity=sum(r['resident_envelope_blocks'] > 4096 for r in rows),
        legacy_full_new_head_charge=table('legacy_cap_feasible'),
        ae_full_cap_same_reserve_and_endpoints=table('ae_full_cap_allowed'),
        limits='Same observed PF states only. Alternate criteria do not run a scheduler. '
               'No alternate trajectory, latency, quality or benefit inference. '
               'Legacy criterion is replayed arithmetically, not its batch1024 implementation. '
               'AE full-cap comparison retains its endpoint convention and 5% reserve. '
               'All historical lengths were causal online inputs; this analysis is offline.',
        decisions=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.cell)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'decisions'}, indent=2))
