"""Rebuild the bounded numerical qualification from retained evidence; no GPU."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAYERS = {f'model.layers.{i}.mlp.experts' for i in range(16)}
WIDTHS = {1, 16, 32, 64, 128, 160}
INSTALLED = {'fused_moe': 'a8015d90908883d3dc459e7a508d2de56bfbdff678c5f36e23d0410ef5a04683',
             'moe_align_block_size': 'c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b'}
NUMERIC = ('finite', 'allclose', 'bit_equal', 'maxabs', 'reference_l2', 'relative_l2', 'atol', 'rtol')


def require(value, label):
    if not value: raise ValueError(label)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical(case, positive=True):
    require(case['finite'] is True and case['allclose'] is positive and case['status'] == 'complete', 'numerical acceptance')
    require(case['atol'] == case['rtol'] == .01 and type(case['bit_equal']) is bool, 'frozen tolerance/bit report')
    for key in ('maxabs', 'reference_l2', 'relative_l2'):
        value = case[key]
        require(value is None and key == 'relative_l2' and case['reference_l2'] == 0
                or type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid numerical scalar ' + key)


def helpers(case, roles=('physical_reference', 'logical')):
    require(case['assignment_restored'] is True, 'assignment hook restoration')
    invocations = case['invocations']
    require([v['role'] for v in invocations] == list(roles), 'invocation roles/order')
    configs = []
    for inv in invocations:
        require(len(inv['helper_calls']) == 1, 'exactly one real assignment helper call')
        h = inv['helper_calls'][0]; physical = inv['role'] == 'physical_reference'; g = 384 if physical else 64
        require(h['global_num_experts'] == g and h['map_shape'] == [g] and h['rows'] == case['rows']
                and h['top_k'] == 8 and h['incoming_ignore_invalid_experts'] is True
                and h['effective_ignore_invalid_experts'] is physical and h['status'] == 'complete', 'actual helper parameters')
        require(type(h['config']['BLOCK_SIZE_M']) is int and h['config']['BLOCK_SIZE_M'] > 0, 'real kernel config')
        configs.append(h['config'])
    require(all(c == configs[0] for c in configs), 'same E384 kernel config')


def analyze(directory):
    result = dict(schema='logical_alignment_analysis_v1', status='UNRUN', issues=[], inputs={},
        scope='CPU evidence join. Tensor outputs were not retained: finite/allclose are original recorded comparisons, not recomputed tensors. No quality or performance claim.')
    try:
        def read(name, base=directory):
            path = base / name; result['inputs'][str(path)] = digest(path)
            return json.loads(path.read_text())
        execution = read('execution.json'); q = directory / 'qualification'
        raw, qual = read('raw.json', q), read('logical_alignment.json', q)
        config, shared = read('config.json', q), read('shared_pool.json', q)
        resources, initial = read('measurement_resources.json', q), read('resources.json', q)
        terminal, workload = read('status.json', q), read('workload.json', q)
        sources, prepared = read('sources.json', ROOT), read('prepared/workload.json', ROOT)
        path = q / 'pager/calls.jsonl'; result['inputs'][str(path)] = digest(path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result['status'] = 'NOT_QUALIFIED'
        require(execution['status'] == 'COMPLETE_NUMERIC_QUALIFICATION' and execution['returncode'] == 0, 'execution incomplete')
        require(raw['status'] == 'COMPLETE' and raw['error'] is None and terminal['status'] == 'COMPLETE', 'native run incomplete')
        require(qual['schema'] == 'logical_alignment_qualification_v1' and qual['hooks_restored'] is True, 'schema/restoration')
        require(config['verify_kernel'] is False and shared['qualification'] is False and shared['mode'] == 'oneshot', 'legacy qualification enabled')
        require(not {'--verify-kernel', '--pool-qualification'} & set(execution['command'] + shared['invocation']), 'legacy flags present')
        require(all(config[k] == v for k, v in dict(requests=5, output_tokens=8, prompt_tokens=128, token_budget=160,
                kv_bytes=1073741824, prefill_limit=32, arrival_interval=0, execution='expert', expert_cap=24).items()), 'configuration differs')
        require(all(workload[k] == prepared[k] for k in ('source_requests', 'actual_prompt_token_ids', 'arrival_traces_s'))
                and len(prepared['source_requests']) == 5, 'prepared workload identity')
        requests = raw['requests']; require(len(requests) == 5 and len({r['request_id'] for r in requests}) == 5, 'request identities/count')
        expected = {r['request_id']: (r, p) for r, p in zip(prepared['source_requests'], prepared['actual_prompt_token_ids'])}
        for r in requests:
            source, prompt = expected[r['request_id']]; times = r['token_times_s']; tokens = r['output_token_ids']
            require(r['document_id'] == source['document_id'] and r['prompt_tokens'] == len(prompt) == 128
                    and r['prompt_token_ids_sha256'] == hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest(), 'request prompt identity')
            require(r['status'] == 'completed' and r['max_output_tokens'] == len(tokens) == len(times) == 8
                    and r['arrival_s'] == 0 and all(math.isfinite(t) for t in times) and times == sorted(times)
                    and r['arrival_s'] <= r['admission_s'] <= times[0] <= times[-1] == r['completion_s'] <= raw['observation_end_s'], 'request completion/token clock')
            events = [e for e in raw['output_events'] if e['request_id'] == r['request_id']]
            require(len(events) == 8 and all(e['prefix_valid'] is True and e['chunk_size'] == 1
                and e['cumulative_token_ids'] == tokens[:i+1] and e['new_token_ids'] == [tokens[i]] and e['received_s'] == times[i]
                and e['finished'] is (i == 7) for i, e in enumerate(events)), 'token event reconstruction')
        require(len(raw['output_events']) == 40 and terminal['requests_completed'] == 5 and terminal['generated_tokens'] == 40, 'five requests/40 outputs')
        for resource in (initial, resources, shared['allocation']):
            require(resource['layer_caps'] == dict.fromkeys(LAYERS, 21) and resource['expert_slots_total'] == 384
                and resource['private_slots_total'] == 336 and resource['shared_slots'] == 48
                and resource['expert_scratch_bytes'] == 4831838208, 'physical expert resources')
            stores = resource['expert_scratch_unique_storages']
            require(len(stores) == len({(s['device'], s['pointer']) for s in stores}) == 2
                    and sum(s['bytes'] for s in stores) == 4831838208, 'unique physical storage')
        require(initial['kv_cache_memory_bytes'] == resources['actual_unique_kv_storage_bytes'] == 1073741824, 'actual 1GiB KV')
        require(len(sources) == 11 and shared['sources'] == sources, 'original source set')
        for name, sha in sources.items():
            require(digest(ROOT / 'source' / name) == qual['source_sha256']['source/' + name] == sha, 'source hash ' + name)
        require(qual['source_sha256']['instrumentation/run_logical_alignment.py'] == digest(ROOT / 'instrumentation/run_logical_alignment.py'), 'wrapper hash')
        require({k: v['sha256'] for k, v in qual['installed_sources'].items()} == INSTALLED, 'installed Python hashes')
        require([r['call_id'] for r in records] == list(range(len(records))), 'whole trace IDs/order')
        measured = [r for r in records if r['measurement']]; actual = qual['actual_calls']
        require(len(actual) == len(measured) > 0 and len({c['call_id'] for c in actual}) == len(actual), 'measurement join counts')
        first_large = {}; actual_index = {}
        for c, r in zip(actual, measured):
            require((c['call_id'], c['layer'], c['context'], c['rows']) == (r['call_id'], r['layer_name'], r['context'], r['rows']), 'actual/trace identity')
            require(r['status'] == 'complete' and r['validation_run'] is False and r['context']['phase'] == 'measurement'
                    and c['returned_path'] == 'logical64_false', 'actual execution/returned path')
            active = sorted({e for row in r['row_topk_experts'] for e in row}); mapping = r['shared_plan']['expert_map_device']; slots = [mapping[e] for e in active]
            require(len(r['row_topk_experts']) == r['rows'] and all(len(row) == len(set(row)) == 8 and all(type(e) is int and 0 <= e < 64 for e in row) for row in r['row_topk_experts']), 'CPU actual route')
            require(len(mapping) == 384 and len(set(slots)) == len(slots) and all(type(s) is int and 0 <= s < 384 for s in slots)
                and c['active_experts'] == r['active_experts'] == active and c['active_physical_slots'] == slots
                and c['cpu_map_valid'] is True and c['actual_shared_weight_identity'] is True, 'active map/weight identity')
            numerical(c); helpers(c); actual_index[c['call_id']] = c
            if c['rows'] >= 160: first_large.setdefault(c['layer'], c)
        prefixes = qual['prefix_cases']; coverage = {layer: sorted(p['rows'] for p in prefixes if p['layer'] == layer) for layer in LAYERS}
        require(len(prefixes) == 96 and all(set(v) == WIDTHS and len(v) == 6 for v in coverage.values()), '16 layers/six actual prefix widths')
        for p in prefixes:
            a = first_large[p['layer']]; numerical(p)
            require(p['call_id'] == a['call_id'] and p['context'] == a['context'] and p['rows'] <= a['rows'], 'first large actual prefix source')
            if p['rows'] == a['rows']:
                require(p['reused_actual_comparison'] is True and not p.get('invocations') and all(p[k] == a[k] for k in NUMERIC), 'actual160 must be reused')
            else: require(not p.get('reused_actual_comparison', False), 'prefix incorrectly reused'); helpers(p)
        require(qual['coverage'] == coverage and qual['missing_coverage'] == {}, 'reported/derived coverage')
        negative = qual['negative_control']; first = actual[0]; original_map = measured[0]['shared_plan']['expert_map_device']; active = first['active_experts']
        rotated = original_map[:64]
        for e, other in zip(active, active[1:] + active[:1]): rotated[e] = original_map[other]
        require(len(active) >= 2 and all(negative[k] == first[k] for k in ('call_id', 'layer', 'context', 'rows', 'active_experts'))
                and negative['rotated_map'] == rotated, 'fixed first-call all-active cyclic negative')
        numerical(negative, False); helpers(negative, ('negative_logical',))
        require(qual['status'] == 'QUALIFIED', 'wrapper terminal differs from reconstructed qualification')
        result.update(status='QUALIFIED', requests=5, output_tokens=40, all_trace_calls=len(records), actual_calls=len(actual),
            prefix_cases=96, coverage=coverage, helper_calls=sum(len(c.get('invocations', [])) for c in [*actual, *prefixes, negative]),
            actual_bit_equal=sum(c['bit_equal'] for c in actual), actual_maxabs=max(c['maxabs'] for c in actual),
            negative={k: negative[k] for k in ('call_id', 'layer', 'rows', *NUMERIC)}, resources=resources,
            process_wall_s=execution['process_wall_s'], observation_end_s=raw['observation_end_s'])
    except (FileNotFoundError, KeyError) as exc:
        result.update(status='UNRUN'); result['issues'].append(f'missing evidence: {exc}')
    except (ValueError, TypeError, IndexError) as exc:
        result.update(status='NOT_QUALIFIED'); result['issues'].append(f'{type(exc).__name__}: {exc}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    result = analyze(args.input_dir.resolve())
    with args.out.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    raise SystemExit(0 if result['status'] == 'QUALIFIED' else 1)
