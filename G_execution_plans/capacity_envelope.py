"""Exact finite-workload logical KV envelope, not a latency simulator.

Assumes one output sequence/request, full attention, no prefix/spec/connector,
zero watermark/lookahead, synchronous scheduling, fresh drained episode.
The bound sums maximum declared per-request capacity regardless of time.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def capacity_regime(comparisons, workloads, required_plans):
    """Diagnostic for the declared candidate set, never an online selector."""
    regimes = {}
    for label in workloads:
        verified = {}
        for plan in required_plans:
            runs = [r for r in comparisons if r['plan'] == plan]
            # A missing, ambiguous or invalid endpoint cannot certify the set.
            if (len(runs) == 1 and runs[0].get('startup_domain_verified')
                    and runs[0]['status'] == 'STARTUP_COMPLETE_NO_SERVICE_RUN'):
                verified[plan] = runs[0]['coverage'][label]['capacity_nonbinding_proven']
        unresolved = [p for p in required_plans if p not in verified]
        if not required_plans or unresolved:
            state = 'UNDETERMINED'
        elif all(verified.values()):
            state = 'EXECUTION_COST_ONLY'
        else:
            state = 'CAPACITY_NOT_EXCLUDED_REQUIRE_SERVICE_EVIDENCE'
        regimes[label] = dict(state=state, unresolved_plans=unresolved,
                             nonbinding_proven_plans=[p for p,v in verified.items() if v],
                             capacity_not_excluded_plans=[p for p,v in verified.items() if not v])
    return dict(required_plans=required_plans, workloads=regimes,
                scope='Only the frozen finite workloads and declared measured plans; no memory monotonicity inference',
                service_capacity_effect_measured=False,
                execution_cost_only_meaning='Rank by measured execution and service costs; no execution winner is established here',
                lower_capacity_meaning='Below the sufficient envelope is not proof of binding capacity or joint optimization value',
                deployment_rule='Joint KV reasoning needs evidence that plan memory changes admission, preemption or supported context',
                future_information='Offline finite-workload diagnostic; not future knowledge available to an online policy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--startup', action='append', type=Path, default=[])
    p.add_argument('--output', type=Path, default=ROOT/'evidence/capacity_envelope.json')
    args = p.parse_args()
    history = json.loads((ROOT/'evidence/historical_feasibility.json').read_text())
    model = json.loads((ROOT/'native_sources/model_config.json').read_text())
    block_size = 16
    bytes_per_token = (model['num_hidden_layers'] * 2 * model['num_key_value_heads'] *
                       (model['hidden_size']//model['num_attention_heads']) * 2)
    bytes_per_block = block_size * bytes_per_token
    result = dict(status='CONDITIONAL_LOGICAL_CAPACITY_CERTIFICATE',
                  measured_service_performance=False, block_size_tokens=block_size,
                  kv_bytes_per_token=bytes_per_token, kv_bytes_per_logical_block=bytes_per_block,
                  assumptions=['fresh episode; only listed requests; one output sequence each',
                               'one full-attention KV group, BF16 TP1',
                               'prefix caching, speculation, KV connector and asynchronous scheduling disabled',
                               'lookahead=0 and watermark=0; no extra retained session or KV copies',
                               'prompt+max_output<=max_model_len; outputs never exceed max_tokens',
                               'one null block excluded from usable pool; finish/preempt frees request blocks'],
                  scope='Exact upper bound on simultaneous logical KV demand under stated semantics; NOT throughput upper bound or observed peak',
                  workloads={}, startup_comparisons=[])
    for label in ('low', 'high'):
        path = ROOT/'inputs'/f'workload_{label}.json'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = history['source_sha256'][f'D_prefill_budget_20261004/workload_{label}.json']
        assert sha == expected, 'Frozen input changed'
        work = json.loads(path.read_text())
        assert len({r['request_id'] for r in work}) == len(work)
        lengths = [len(r['prompt_token_ids'])+r['max_tokens'] for r in work]
        assert all(0 < n <= 4096 for n in lengths)
        assert all(isinstance(r['max_tokens'], int) and r['max_tokens'] > 0 for r in work)
        blocks = sum((n+block_size-1)//block_size for n in lengths)
        historical_usable = history['workloads'][label]['usable_kv_blocks']
        result['workloads'][label] = dict(input_sha256=sha, requests=len(work),
            total_declared_tokens=sum(lengths), max_request_length=max(lengths),
            all_requests_max_blocks=blocks, all_requests_max_kv_gib=blocks*bytes_per_block/2**30,
            min_sufficient_total_blocks_including_null=blocks+1,
            historical_usable_blocks=historical_usable,
            historical_margin_blocks=historical_usable-blocks,
            historical_margin_gib=(historical_usable-blocks)*bytes_per_block/2**30,
            historical_status='old D GPU/config; conditional comparison, not current plan measurement')
    for path in args.startup:
        run = json.loads(path.read_text())
        record = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      plan=run.get('plan'), status=run['status'])
        if run['status'] == 'STARTUP_COMPLETE_NO_SERVICE_RUN':
            domain = run.get('observed_capacity_domain', {})
            expected = dict(num_lookahead_tokens=0, watermark_blocks=0,
                            enable_prefix_caching=False, async_scheduling=False,
                            has_kv_transfer_config=False, has_speculative_config=False,
                            num_kv_cache_groups=1, scheduler_block_size_tokens=16, model_type='olmoe')
            verified = (all(domain.get(k) == v for k,v in expected.items()) and
                        domain.get('kv_cache_spec_types') == ['FullAttentionSpec'])
            usable = run['scheduler_free_kv_blocks']
            assert usable == run['scheduler_kv_blocks_including_reserved_block']-1
            record.update(startup_domain_verified=verified, observed_domain=domain, usable_blocks=usable,
                          coverage={k: dict(margin_blocks=usable-v['all_requests_max_blocks'],
                                           capacity_nonbinding_proven=bool(verified and usable>=v['all_requests_max_blocks']))
                                    for k,v in result['workloads'].items()})
        result['startup_comparisons'].append(record)
    frozen = json.loads((ROOT/'evidence/review_endpoint_config.json').read_text())
    result['capacity_first_decision'] = capacity_regime(
        result['startup_comparisons'], result['workloads'], frozen['plans'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
