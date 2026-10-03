"""Print conditional pre-execution predictions from the retained ledger counts.

The original D/E raw and A's old commit-recheck receipt are absent from this
checkout.  This script must not label count-only certificates as legal native
actions or measured time savings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_model import count_only_prediction


def main():
    events = {
        'D859': count_only_prediction(193, 95, 233),
        'E1399': count_only_prediction(277, 106, 184),
    }
    print(json.dumps({
        'status': 'LEDGER_COUNT_ONLY_CONDITIONAL_UNRUN',
        'source': 'RESULT_LEDGER.md 2026-09-15 commit recheck and cost entries',
        'events': events,
        'prediction_if_all_native_gates_hold': (
            'DIRECT_RESUME keeps the planned victim resident; the custom '
            'forced-preemption receipt for that victim is absent, while the '
            'same target should deliver its next output no later if both '
            'branches remain live and produce one. Early EOS is recorded '
            'separately. Any native '
            'replacement preemption or later peer delay remains observable '
            'and is charged to the direct branch.'),
        'falsifiers': [
            'a required slot, queue, ownership, or connector gate is false',
            'the target first new output is delayed versus the paired old commit',
            'native pressure immediately evicts the retained victim or another peer',
            'complete-request cost erases the local benefit',
        ],
        'full_request_experiment': 'UNRUN',
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
