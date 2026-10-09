"""Only donor-ranking and actual child wiring risks; no model or GPU."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import exchange_max_release as policy
import run_cell_max_release as child

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--record', type=Path, default=ROOT/'session-after-failure-20261009-r01/cell-01-cap256-exchange_once/output/recovery-capacity-exchange.json',
                    help='Existing C1 decision JSON; read-only endpoint fixture')
args = parser.parse_args()
source = inspect.getsource(policy.PARENT.FROZEN._attach)
adapted = policy._max_source(source)
assert adapted.replace(policy.NEW_SELECTION, policy.OLD_SELECTION).replace(
    policy.NEW_DECISION, policy.OLD_DECISION) == source

# Reuse only the already-frozen failure-trigger fixture definitions.
path = ROOT/'check_after_failure_cpu.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'aa3a180e7630629a417af2abe79e62d0c2446371f4358abdac98859fd815e258'
tree = ast.parse(path.read_text())
tree.body = tree.body[:next(i for i, n in enumerate(tree.body) if isinstance(n, ast.With))]
ns = dict(__file__=str(path))
exec(compile(tree, str(path)+'[fixture-only]', 'exec'), ns)
ns['policy'] = NS(**dict(policy.PARENT.__dict__, install=policy.install))
with patch.object(policy.PARENT.FROZEN.time, 'perf_counter', lambda:100.):
    for mode in ('stall8', 'exchange_once'):
        c = ns['fixture'](mode)
        c.s.schedule(); ns['fail'](c); out = c.s.schedule()
        decision = next(e for e in c.data['events'] if e['kind'] == 'decision')
        assert decision['minimum_cost_donor'] == 'donor'
        assert decision['maximum_release_donor'] == decision['donor'] == 'larger'
        assert {r['request'] for r in decision['donors'] if r['capacity']['capacity_fit']} == {'donor', 'larger'}
        assert c.data['action_count'] == int(mode == 'exchange_once')
        assert out.preempted_req_ids == (['larger'] if mode == 'exchange_once' else [])
        assert c.alloc_calls == 1 and c.data['failure_trigger']['first_failure']['latched']
        assert c.data['max_release_adapter']['frozen_after_failure_sha256'] == policy.PARENT_SHA
        if mode == 'exchange_once':
            assert c.s._rotation_forced_count == 1 and c.bigger.status.name == 'PREEMPTED'
            assert c.pool.free == 12 and c.d in c.s.running
            c.receipts[c.t.request_id] = 99.; c.s.schedule()
            assert next(e for e in c.data['events'] if e['kind'] == 'release')['reason'] == 'FIRST_CLIENT_RECEIPT'
        ns['close'](c)

# Execute the exact replacement selection statements, including original shadow.
choose_ns = {}
selection = '\n'.join(line[8:] for line in policy.NEW_SELECTION.splitlines())
exec('def choose(sufficient):\n'+''.join('    '+line+'\n' for line in selection.splitlines())+
     '    return minimum_cost_row, donor_row\n', choose_ns)
choose = choose_ns['choose']
def row(rid, release, host, index):
    return dict(request=rid, immediate_releasable_blocks=release,
                host_missing_materialized_blocks=host, running_index=index,
                release_excess_blocks=release-10)
rows = [row('smaller', 10, 0, 9), row('host-costlier', 12, 4, 9),
        row('earlier', 12, 3, 1), row('tail', 12, 3, 2)]
assert tuple(r['request'] for r in choose(rows)) == ('smaller', 'tail')

# Recorded C1 decision only: same qualified/sufficient set, not a replay result.
record = args.record
decision = next(e for e in json.loads(record.read_text())['events'] if e['kind'] == 'decision')
sufficient = [r for r in decision['donors'] if r['capacity']['capacity_fit']]
minimum, maximum = choose(sufficient)
assert minimum['request'] == decision['donor'] == 'measured/b-normal-0056344-short-a326ef9f'
assert maximum['request'] == 'measured/b-normal-0054735-long-b44a76d0'
assert (maximum['immediate_releasable_blocks'], maximum['host_missing_materialized_blocks'],
        maximum['computed_tokens'], maximum['output_tokens'], maximum['capacity']['margin_blocks']) == (253, 252, 4040, 969, 170)
child.self_check()
print('PASS: original sufficient set, max-release/Host/tail ranking, both actual trigger modes, one native preempt, Q1/uninstall, real C1 endpoint and exact child source adaptation. GPU_UNRUN.')
