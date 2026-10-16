"""Freeze W2-selected settings and prepare the two-arm W2-2lambda plan.

This only prepares a local plan; it never submits GPU work. W1/W3 reuse the
same selected arms from their existing frontiers, without per-workload tuning.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'figures' / 'F2'))
from plot_corrected import read_panel


def prepare(plan_path, code_snapshot, cap, ha, reason, session_name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', session_name):
        raise ValueError('Use a new session basename, without path separators')
    if not reason.strip():
        raise ValueError('Explain the W2 selection and its service tradeoff')
    panel = read_panel(plan_path, code_snapshot)
    if panel['workload'] != 'W2':
        raise ValueError('Selection must use W2 development results')
    if panel['missing']:
        raise ValueError('W2 frontier is unmeasured/incomplete; no settings frozen')
    source = json.loads(plan_path.read_text())
    receipt = json.loads((ROOT / 'runs' / panel['session'] / 'receipt.json').read_text())
    if receipt['status'] != 'COMPLETE':
        raise ValueError('W2 group must have completed')
    if session_name == panel['session'] or (ROOT / 'runs' / session_name).exists():
        raise ValueError('Use an unused session; never overwrite an experiment')
    selected = []
    for label in (f'cap{cap}', f'H={ha}'):
        matches = [p for p in panel['points'] if p['label'] == label]
        if len(matches) != 1:
            raise ValueError(f'Selected arm is absent or ambiguous: {label}')
        selected.append(matches[0])
    plan = copy.deepcopy(source)
    plan['session'] = str(Path(source['session']).parent / session_name)
    plan['cells'] = [copy.deepcopy(p['config']) for p in selected]
    for cell in plan['cells']:
        cell['arrival_rate'] = 2 * panel['arrival_rate']
    plan['provenance']['transfer_selection'] = dict(
        source_session=panel['session'], code_snapshot=code_snapshot,
        plan_sha256=panel['plan_sha256'],
        metrics_sha256=panel['metrics_sha256'],
        receipt_sha256=panel['receipt_sha256'],
        reason=reason.strip(), selected_points=selected,
        semantics='W2 development selection, reused for W1/W3 and W2-2lambda. '
        'No retuning on transfer results; one run per point is exploratory. '
        'The source W2 points are selection data, not independent confirmation.')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w2-plan', type=Path, required=True)
    parser.add_argument('--code-snapshot', required=True)
    parser.add_argument('--cap', type=int, required=True)
    parser.add_argument('--ha', type=int, choices=(16, 64, 256, 1024), required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--session', required=True, help='New remote session basename')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = prepare(args.w2_plan, args.code_snapshot, args.cap, args.ha,
                       args.reason, args.session)
        # Serialize before opening: invalid data cannot leave a partial plan.
        payload = json.dumps(plan, indent=2, allow_nan=False) + '\n'
        with args.output.open('x') as stream:
            stream.write(payload)
    except (ValueError, AssertionError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(dict(status='PREPARED_NOT_SUBMITTED', output=str(args.output),
                          session=plan['session'], cells=len(plan['cells']))))


if __name__ == '__main__':
    main()
