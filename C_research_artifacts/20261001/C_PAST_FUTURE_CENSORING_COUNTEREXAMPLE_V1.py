"""Synthetic metadata-loss example; not an observed state or new controller."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from C_PAST_FUTURE_AE_PREDICTOR_V1 import (
    AERequest, PastFutureAEPredictor, normal_req_tuple, future_peak_tokens)


def example():
    running = [AERequest(f'r{i}', 2700, 100, 1024, 'RUNNING') for i in range(17)]
    head = AERequest('head', 3000, 0, 1024, 'WAIT_IN_QUEUE')
    cases = []
    for finish, historical_cap in [('length', 128), ('stop', 1024)]:
        predictor = PastFutureAEPredictor(1024, 20261001)
        for i in range(40):
            predictor.record_completed_measured(f'old{i}', 128,
                phase='measured', finish_reason=finish)
        prediction = predictor.score_waiting(running, head, capacity_tokens=65536)
        cases.append(dict(historical_output_tokens=128, historical_records=40,
            hypothetical_historical_request_cap=historical_cap,
            finish_reason=finish, history_used=list(predictor.history()),
            prediction=asdict(prediction)))
    full_cap = future_peak_tokens([normal_req_tuple(r, r.max_output_tokens)
                                  for r in running + [head]])
    assert cases[0]['prediction'] == cases[1]['prediction']
    assert cases[0]['prediction']['fits'] and full_cap == 67215 > 65536
    assert cases[0]['prediction']['worst_peak_tokens'] == 51087
    return dict(schema='c-past-future-censoring-counterexample-v1',
        status='SYNTHETIC_CPU_ONLY_UNSELECTED_BACKUP',
        running=[asdict(r) for r in running], head=asdict(head), cases=cases,
        full_cap_joint_ae_peak_tokens=full_cap,
        scope='Two histories differing in finish reason have identical predictions. '
              'The length case is compatible with unobserved longer natural outputs; '
              'the stop case actually observed termination. Historical request cap '
              'is explanatory metadata, unavailable in the existing history API. '
              'Current GPU experiments have a common cap1024 and are unchanged. '
              'No actual future EOS, observed overload, native action, service gain, '
              'hard-guarantee violation, censoring estimator or novelty is claimed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = example()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(result['status'], result['cases'][0]['prediction'],
          'full_cap_peak', result['full_cap_joint_ae_peak_tokens'])
