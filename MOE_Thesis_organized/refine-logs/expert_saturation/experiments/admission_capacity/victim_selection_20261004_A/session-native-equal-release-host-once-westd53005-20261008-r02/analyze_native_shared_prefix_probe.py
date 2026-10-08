#!/usr/bin/env python3
"""One native-tail shared-prefix characterization; physical-release shadows only."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import analyze_native_group as base
import analyze_native_mixed_budget_probe as mixed


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_shared_prefix_probe_r01'
ROLE = 'SHARED_PREFIX_NATIVE_TAIL_CHARACTERIZATION'
SEMANTICS = ('One synthetic shared-article tail run, not a policy comparison. Native suffix legality '
    'is distinct from the inherited pure-decode qualified flag. Max-release choices are read-only '
    'shadows, not executed victims or measured gains. Immediate physical release is compared with '
    'the actual native free-pool increment; held pages are not assumed reclaimable. Events and '
    'repeated candidate observations are not independent runs. Missing fields remain UNKNOWN.')


def input_check(config, workload, raw):
    rows, prompts = workload.get('source_requests', []), workload.get('actual_prompt_token_ids', [])
    ids = [row.get('request_id') for row in rows]
    meta = config.get('shared_prefix_diagnostic', {})
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[row.get('prompt_length_stratum')].append(i)
    expected_pairs = []
    shape = (len(rows) == len(prompts) == len(set(ids)) == 320
             and len(strata) == 64 and all(len(v) == 5 for v in strata.values()))
    if shape:
        expected_pairs = [dict(stratum=s, first_index=v[0], second_index=v[1],
            first_request=ids[v[0]], second_request=ids[v[1]], prompt_token_count=len(prompts[v[0]]))
            for s, v in sorted(strata.items())]
    pairs_match = shape and meta.get('pairs') == expected_pairs
    token_hashes = [hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
                    for tokens in prompts]
    multiplicities = Counter(Counter(token_hashes).values())
    checks = dict(input_shape=shape, metadata_matches_workload=meta == workload.get('shared_prefix_diagnostic'),
        metadata_kind=meta.get('kind') == 'SYNTHETIC_SHARED_ARTICLE_OPPORTUNITY_DIAGNOSTIC',
        declared_counts=all(meta.get(k) == v for k, v in dict(pair_count=64, paired_requests=128,
            unpaired_requests=192, unique_documents=256).items()),
        pairs_follow_first_two_per_stratum=pairs_match,
        paired_prompts_identical=pairs_match and all(prompts[p['first_index']] == prompts[p['second_index']]
            and rows[p['first_index']]['prompt'] == rows[p['second_index']]['prompt']
            and rows[p['first_index']]['document_id'] == rows[p['second_index']]['document_id']
            for p in expected_pairs),
        token_multiplicities=multiplicities == {1: 192, 2: 64},
        document_count=len({r.get('document_id') for r in rows}) == 256,
        prompt_hashes=shape and all(r.get('prompt_token_ids_sha256') == h
            and r.get('prompt_token_count') == len(tokens)
            and r.get('prompt_sha256') == hashlib.sha256(r['prompt'].encode()).hexdigest()
            for r, tokens, h in zip(rows, prompts, token_hashes)),
        external_arrival_trace=workload.get('arrival_traces_s') == {'steady': [i / 100 for i in range(320)]},
        input_prefix_enabled=config.get('enable_prefix_caching') is True,
        workload_digest=config.get('workload_sha256') == base.digest(workload))
    actual = {r.get('request_id'): r for r in raw.get('requests', [])} if raw is not None else {}
    measured = raw is not None and len(raw.get('requests', [])) == len(actual) == 320 and set(actual) == set(ids)
    checks['measured_request_ids'] = measured if raw is not None else None
    checks['measured_arrivals_and_prompts'] = (measured and all(
        actual[rid].get('arrival_s') == i / 100
        and actual[rid].get('prompt_token_ids_sha256') == token_hashes[i]
        and actual[rid].get('prompt_tokens') == len(prompts[i]) for i, rid in enumerate(ids))) if raw is not None else None
    return dict(status='VERIFIED' if all(v is True for v in checks.values()) else 'MISMATCH_OR_MISSING',
        checks=checks, prompt_multiplicity_counts=dict(multiplicities),
        semantics='Input copies do not prove runtime sharing. Raw prompt hashes and external arrival times are checked per request; budgets are checked separately.')


def domain_check(cell, store):
    config, engine = cell.get('config') or {}, cell.get('engine_args') or {}
    checks = dict(measured_config_prefix=config.get('enable_prefix_caching') is True,
        engine_prefix=engine.get('enable_prefix_caching') is True,
        native_manager_prefix=store.get('prefix_caching') is True,
        coordinator=store.get('coordinator_type') == 'UnitaryKVCacheCoordinator',
        inherited_q1_configuration=cell.get('configuration_check', {}).get('status') == 'VERIFIED')
    return dict(status='VERIFIED' if all(checks.values()) else 'MISMATCH_OR_MISSING', checks=checks,
        observed=dict(config_prefix=config.get('enable_prefix_caching'), engine_prefix=engine.get('enable_prefix_caching'),
                      manager_prefix=store.get('prefix_caching'), coordinator_type=store.get('coordinator_type')),
        semantics='Inherited configuration checks retain native_full, Q1, funding tail, admission and other private gates. Prefix enabling and Unitary coordinator are the declared runtime-domain changes.')


def load_shadow(source, executed_sha):
    if source is None or not source.is_file():
        return None, None
    payload = source.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    if source_sha != executed_sha:
        return None, source_sha
    node = next((n for n in ast.parse(payload).body
                 if isinstance(n, ast.FunctionDef) and n.name == '_max_release_shadow_choice'), None)
    if node is None:
        return None, source_sha
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[node.name], source_sha


def physical_known(row):
    return (all(base.count_known(row.get(k)) for k in
                ('held_blocks', 'immediate_releasable_blocks', 'shared_blocks'))
            and row.get('release_state_error', 'NOT_RECORDED') is None
            and row['immediate_releasable_blocks'] + row['shared_blocks'] <= row['held_blocks'])


def shared_observations(cell, store, raw, source, executed_sha):
    decisions = store.get('victim_decisions')
    choose, source_sha = load_shadow(source, executed_sha)
    if not isinstance(decisions, list):
        return dict(status='OBSERVATIONS_UNAVAILABLE', decisions=None, semantics=SEMANTICS)
    records, counts, shadow_times = [], Counter(), []
    joins = {j['decision_index']: j for j in cell.get('native_victim_observation', {}).get(
        'actual_release', {}).get('joins', [])}
    for i, event in enumerate(decisions):
        rows = event.get('candidates', [])
        indices = [r.get('index') for r in rows]
        valid_suffix = (bool(rows) and all(type(v) is int for v in indices)
            and indices == list(range(indices[0], indices[-1] + 1))
            and type(event.get('unprocessed_suffix_start')) is int
            and indices[0] == event['unprocessed_suffix_start']
            and all(isinstance(r.get('request'), str) for r in rows)
            and len({r['request'] for r in rows}) == len(rows)
            and rows[-1].get('request') == event.get('native_tail'))
        phase = event.get('active_protection_or_phase')
        tail = rows[-1] if valid_suffix else {}
        record = dict(decision_index=i, step=event.get('step'), valid_native_suffix=valid_suffix,
            active_protection_or_phase=phase, actual_tail_verified=event.get('rule') == 'tail'
                and event.get('changed') is False and event.get('selected') == event.get('native_tail'),
            same_held_alternatives=None, same_held_different_release=None,
            same_held_greater_release=None, same_held_unknown_release=None)
        counts['candidate_observations'] += len(rows)
        counts['known_physical_observations'] += sum(physical_known(r) for r in rows)
        counts['unknown_physical_observations'] += sum(not physical_known(r) for r in rows)
        counts['pure_decode_unqualified_observations'] += sum(r.get('qualified') is False for r in rows)
        counts['shared_candidate_observations'] += sum(physical_known(r) and r['shared_blocks'] > 0 for r in rows)
        if valid_suffix:
            counts['valid_native_suffix_decisions'] += 1
            counts['suffix_with_observed_shared_blocks'] += any(physical_known(r) and r['shared_blocks'] > 0 for r in rows)
        if valid_suffix and phase is False:
            counts['unprotected_native_suffix_decisions'] += 1
            counts['unprotected_suffix_with_shared_blocks'] += any(physical_known(r) and r['shared_blocks'] > 0 for r in rows)
            if base.count_known(tail.get('held_blocks')):
                equal = [r for r in rows[:-1] if r.get('held_blocks') == tail['held_blocks']]
                known = [r for r in equal if physical_known(r) and physical_known(tail)]
                record.update(same_held_alternatives=len(equal),
                    same_held_unknown_release=len(equal) - len(known),
                    same_held_different_release=sum(r['immediate_releasable_blocks'] != tail['immediate_releasable_blocks'] for r in known),
                    same_held_greater_release=sum(r['immediate_releasable_blocks'] > tail['immediate_releasable_blocks'] for r in known))
        shadow = event.get('max_release_shadow')
        if not valid_suffix or type(phase) is not bool:
            record['shadow_check'] = 'UNKNOWN_NATIVE_SUFFIX_OR_GUARD'
        elif choose is None:
            record['shadow_check'] = 'EXECUTED_SOURCE_UNAVAILABLE'
        else:
            proposed, reason = choose(rows, phase)
            selected = next(r for r in rows if r['index'] == proposed)
            expected = dict(mode='SHADOW', proposed_request=selected['request'],
                changed_from_tail=selected['request'] != event['native_tail'], action_requested=False,
                fallback=reason, tail_releasable_blocks=tail.get('immediate_releasable_blocks'),
                proposed_releasable_blocks=selected.get('immediate_releasable_blocks'))
            record['shadow_check'] = ('MATCH' if isinstance(shadow, dict) and all(
                k in shadow and shadow[k] == value and (type(value) is not bool or type(shadow[k]) is bool)
                for k, value in expected.items()) else 'MISMATCH')
            record['recomputed_shadow'] = expected
        value = shadow.get('selector_wall_s') if isinstance(shadow, dict) else None
        if base.finite(value) and value >= 0:
            shadow_times.append(value)
        join = joins.get(i, {})
        chosen = [r for r in rows if r.get('request') == event.get('selected')]
        immediate = chosen[0]['immediate_releasable_blocks'] if len(chosen) == 1 and physical_known(chosen[0]) else None
        actual = join.get('actual_released_blocks')
        comparison = ('UNKNOWN' if join.get('join_status') != 'UNIQUE' or immediate is None
            or not base.count_known(actual) or join.get('free_counter_delta_consistent') is not True
            else 'MATCH' if immediate == actual else 'MISMATCH')
        # Augment the inherited unique join rather than duplicating the raw event.
        join.update(selected_immediate_releasable_blocks=immediate,
                    actual_vs_immediate_release=comparison,
                    actual_minus_immediate_blocks=actual-immediate if comparison != 'UNKNOWN' else None)
        record['actual_vs_immediate_release'] = comparison
        records.append(record)
    actual_release = cell.get('native_victim_observation', {}).get('actual_release', {})
    counts['actual_changed_decisions'] = sum(e.get('selected') is not None and e.get('native_tail') is not None
        and e['selected'] != e['native_tail'] for e in decisions)
    counts['shadow_changed_decisions'] = sum(r.get('recomputed_shadow', {}).get('changed_from_tail') is True for r in records)
    counts['same_held_alternative_decisions'] = sum((r['same_held_alternatives'] or 0) > 0 for r in records)
    counts['same_held_different_release_decisions'] = sum((r['same_held_different_release'] or 0) > 0 for r in records)
    counts['same_held_greater_release_decisions'] = sum((r['same_held_greater_release'] or 0) > 0 for r in records)
    counts['same_held_unknown_release_pairs'] = sum(r['same_held_unknown_release'] or 0 for r in records)
    duration = raw.get('observation_end_s') if raw else None
    total = sum(shadow_times) if shadow_times else None
    verified = (raw is not None and choose is not None and actual_release.get('status') == 'ANALYZED'
        and not actual_release.get('actual_without_unique_native_decision')
        and all(r['actual_tail_verified'] and r['shadow_check'] == 'MATCH'
                and r['actual_vs_immediate_release'] == 'MATCH' for r in records))
    return dict(status='VERIFIED' if verified else 'MISMATCH_OR_MISSING', decisions=len(decisions),
        source=str(source) if source else None, source_sha256=source_sha, executed_source_sha256=executed_sha,
        source_verified=choose is not None, counts=dict(counts),
        shadow_check_counts=dict(Counter(r['shadow_check'] for r in records)),
        fallback_counts=dict(Counter(r.get('recomputed_shadow', {}).get('fallback', 'UNKNOWN') for r in records)),
        actual_release_check_counts=dict(Counter(r['actual_vs_immediate_release'] for r in records)),
        max_release_shadow_cost=dict(recorded_decisions=len(shadow_times), unknown_decisions=len(decisions)-len(shadow_times),
            total_recorded_s=total, distribution=base.distribution(shadow_times),
            fraction_capture=total/duration if total is not None and base.finite(duration) and duration > 0 else None,
            semantics='Shadow-only ranking time, contained in outer selector_wall_s. Candidate observation and shadow costs must not be added to that total.'),
        per_decision=records, semantics=SEMANTICS)


def analyze(session, policy_source, input_config):
    plan = base.read(session / 'plan.json')
    cells = plan.get('cells', [])
    if (plan.get('kind') != 'PRO6000_NATIVE_EQUAL_HELD_ONCE' or plan.get('experiment_role') != ROLE
            or len(cells) != 1 or cells[0].get('label') != 'shared-prefix-tail'
            or cells[0].get('native_victim_rule') != 'tail' or cells[0].get('requests') != 320
            or cells[0].get('input_case') != 'high'):
        raise ValueError('Expected the declared single shared-prefix-tail characterization plan')
    spec = cells[0]
    directory = session / ('cell-00-' + spec['label'])
    cell, raw = base.read_native_cell(directory, spec, plan, plan.get('configuration', {}).get('max_seconds', 600), policy_source)
    archive = base.archive_dir(directory)
    store = base.optional(archive / 'selective-store.json', {})
    environment = base.optional(archive / 'environment.json', {})
    executed = directory / 'candidate_native_oldest_strong_r01/pkg'
    source = executed / 'staged_store_rotation.py'
    source = source if source.is_file() else policy_source
    config_path = executed / 'inputs/pro_high/config.json'
    executed_input = config_path.is_file()
    config_path = config_path if executed_input else input_config
    budget = inputs = dict(status='INPUT_UNAVAILABLE')
    if config_path.is_file() and config_path.with_name('workload.json').is_file():
        config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
        budget = mixed.budget_check(config, workload, cell.get('config') or {}, raw)
        inputs = input_check(config, workload, raw)
        inputs.update(input_config_path=str(config_path), input_config_sha256=mixed.sha(config_path),
            input_workload_sha256=mixed.sha(config_path.with_name('workload.json')),
            input_source='EXECUTED_CELL_PACKAGE' if executed_input else 'SUPPLIED_PACKAGE_FALLBACK')
    warmup = mixed.warmup_check(archive, cell.get('engine_args', {}))
    domain = domain_check(cell, store)
    shared = shared_observations(cell, store, raw, source,
        environment.get('source_sha256', {}).get('staged_store_rotation.py'))
    complete = cell.get('status') == 'COMPLETE' and all(value.get('status') == 'VERIFIED'
        for value in (budget, inputs, warmup, domain, shared))
    return dict(status='COMPLETE_CHARACTERIZATION' if complete else
        'NO_MEASUREMENT' if raw is None else 'INCOMPLETE_OR_INVALID', experiment_role=ROLE,
        session=str(session), plan=plan, receipt=base.optional(session / 'receipt.json'), cell=cell,
        budget_consistency=budget, input_consistency=inputs, fixed_warmup_check=warmup,
        shared_prefix_domain=domain, shared_prefix_observations=shared,
        analysis_code_sha256={Path(module.__file__).name: mixed.sha(Path(module.__file__)) for module in (base, mixed)}
            | {Path(__file__).name: mixed.sha(Path(__file__))}, semantics=SEMANTICS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--policy-source', type=Path, default=PACKAGE / 'pkg/staged_store_rotation.py')
    parser.add_argument('--input-config', type=Path, default=PACKAGE / 'pkg/inputs/pro_high/config.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.session.resolve(), args.policy_source.resolve(), args.input_config.resolve())
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output)), ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
