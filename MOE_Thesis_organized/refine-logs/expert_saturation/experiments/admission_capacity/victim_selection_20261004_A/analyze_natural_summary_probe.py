#!/usr/bin/env python3
"""One natural-summary feasibility capture; stdout or one exclusively created JSON."""
import argparse
import ast
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import analyze_native_cacheopt_cap_bucket_group as policies
import analyze_native_shared_prefix_profiled as profiled


base, mixed = policies.base, policies.mixed
HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_natural_summary_probe_r01/pkg'
KIND, ROLE = 'PRO6000_NATIVE_NATURAL_SUMMARY_PROBE', 'NATURAL_SUMMARY_FEASIBILITY'
MODEL = 'allenai/OLMoE-1B-7B-0924-Instruct'
SEMANTICS = dict(
    evidence='One measured cell after a fresh same-group profile; no independent repeat or strategy comparison.',
    clock='External arrival to host engine-return output/completion. Max gap excludes TTFT and is undefined with fewer than two distinct output timestamps.',
    work='Actual output tokens and termination are reported; the cap is an upper bound, not predicted remaining work.',
    domain='Instruct weights, chat template, task, prompt lengths, EOS behavior, compilation/cache and warmup differ from the base continuation domain. Do not compare their wall times as policy effects.',
    shadows='Alternative suggestions on the executed remaining-budget trajectory are not executed alternatives or causal benefits.',
    hindsight='Candidate final EOS is a retrospective diagnostic, unavailable to the online selector.',
    quality='Eight input-preselected articles, manual review only. No automatic pass, gold summaries, or policy quality-equivalence claim.',
    slo='No application SLO. No goodput threshold is selected or reported.')


def validate_plan(plan):
    cells = plan.get('cells', [])
    if (plan.get('kind') != KIND or plan.get('experiment_role') != ROLE or len(cells) != 2
            or 'pinned_gpu_kv_bytes' in plan):
        raise ValueError('Requires the natural-summary fresh-profile, single-measurement plan')
    for spec, label, profile, requests in zip(cells, ('profile', 'remaining-budget'), (True, False), (0, 320)):
        wanted = dict(label=label, profile_only=profile, requests=requests, input_case=None,
            native_victim_rule='remaining_budget', funding_victim_rule='tail')
        # Profiling does not execute a victim rule; either inherited tail or the measured rule is valid.
        if profile:
            wanted.pop('native_victim_rule')
        if any(key not in spec or spec[key] != value or type(spec[key]) is not type(value)
               for key, value in wanted.items()):
            raise ValueError('Requires profile cell00 and remaining-budget 320-request cell01')
    return cells


def input_check(config, workload, cell, raw):
    sources = workload.get('source_requests', [])
    ids = [row.get('request_id') for row in sources]
    prompts = workload.get('actual_prompt_token_ids', [])
    arrivals = workload.get('arrival_traces_s', {}).get('steady', [])
    task = workload.get('task', {})
    quality_ids = task.get('quality_check_ids', [])
    measured, engine = cell.get('config') or {}, cell.get('engine_args') or {}
    checks = dict(model=config.get('model', {}).get('id') == task.get('model') == MODEL,
        natural_stop=config.get('ignore_eos') is False and config.get('min_tokens') == 0,
        common_cap=config.get('output_tokens') == config.get('max_output_tokens') == 981
            and not config.get('output_tokens_by_request'),
        cohort=len(ids) == len(set(ids)) == len(prompts) == len(arrivals) == 320,
        quality_preselected=len(quality_ids) == len(set(quality_ids)) == 8 and set(quality_ids) <= set(ids),
        source_contract=config.get('workload_sha256') == base.digest(workload),
        measured_model=measured.get('model') == config.get('model'),
        measured_natural_stop=measured.get('ignore_eos') is False and measured.get('min_tokens') == 0,
        measured_cap=measured.get('output_tokens') == 981 and not measured.get('output_tokens_by_request'),
        prefix_off=measured.get('enable_prefix_caching', False) is False
            and engine.get('enable_prefix_caching') is False)
    expected = {row['request_id']: (row.get('prompt_token_ids_sha256'), len(tokens), arrival, 981)
        for row, tokens, arrival in zip(sources, prompts, arrivals)}
    records = raw.get('requests') if raw is not None else None
    checks['recorded_input_identity'] = isinstance(records, list) and len(records) == 320 and {
        row.get('request_id'): (row.get('prompt_token_ids_sha256'), row.get('prompt_tokens'),
            row.get('arrival_s'), row.get('max_output_tokens')) for row in records} == expected
    return dict(status='VERIFIED' if all(checks.values()) else 'MISMATCH_OR_MISSING', checks=checks,
        task=task, common_cap=981, prompt_tokens=base.distribution([len(p) for p in prompts]))


def stop_contract(config, eos, source, expected_sha):
    verified = source.is_file() and mixed.sha(source) == expected_sha
    calls = [node for node in ast.walk(ast.parse(source.read_bytes()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'SamplingParams'] if verified else []
    no_extra_stop = len(calls) == 1 and all(k.arg is not None and k.arg not in
        ('stop', 'stop_token_ids') for k in calls[0].keywords)
    ids = eos.get('hf_eos_token_id') if eos.get('status') == 'READ' else None
    ids = ids if isinstance(ids, list) else [ids]
    ids = sorted({value for value in ids if type(value) is int and value >= 0})
    defaults = [eos.get(key) for key in ('effective_sampling_defaults', 'generation_overrides')]
    defaults_known = all(isinstance(row, dict) for row in defaults)
    no_model_stops = defaults_known and all(row.get(key) in (None, [], '')
        for row in defaults for key in ('stop', 'stop_token_ids'))
    generation_eos = (eos.get('generation_config') or {}).get('eos_token_id')
    gen_ids = generation_eos if isinstance(generation_eos, list) else [generation_eos]
    compatible = generation_eos is None or set(gen_ids) <= set(ids)
    only_eos = (verified and no_extra_stop and no_model_stops and compatible and eos.get('status') == 'READ'
        and ids == [50279] and config.get('ignore_eos') is False and config.get('min_tokens') == 0)
    return dict(status='VERIFIED_EOS_ONLY_STOP_CONTRACT' if only_eos else 'UNKNOWN_STOP_CONTRACT',
        eos_token_ids=ids, default_eos_only=only_eos, request_source_verified=verified,
        no_explicit_request_stops=no_extra_stop, resolved_metadata=eos,
        semantics='A completed stop with null native_stop_reason can be EOS under the verified EOS-only contract. Null alone is not EOS evidence; explicit stopping tokens and final output tokens are retained separately.')


def classify_stop(row, contract):
    tokens, cap = row.get('output_token_ids'), row.get('max_output_tokens')
    known = isinstance(tokens, list) and all(type(token) is int for token in tokens)
    count = len(tokens) if known else None
    final = tokens[-1] if known and tokens else None
    native, finish = row.get('native_stop_reason'), row.get('stop_reason')
    evidence, label = None, 'NOT_COMPLETED'
    if row.get('status') == 'completed':
        if finish == 'length':
            label = 'LENGTH'
        elif finish == 'stop':
            ids = contract['eos_token_ids']
            if type(native) is int and native in ids:
                label, evidence = 'VERIFIED_EOS', 'EXPLICIT_NATIVE_STOP_TOKEN'
            elif final in ids and native is None:
                label, evidence = 'VERIFIED_EOS', 'FINAL_OUTPUT_EOS_TOKEN'
            elif ('native_stop_reason' in row and native is None and contract['default_eos_only']):
                label, evidence = 'VERIFIED_EOS', 'VERIFIED_EOS_ONLY_STOP_CONTRACT'
            else:
                label = 'STOP_UNCLASSIFIED'
        else:
            label = 'UNKNOWN_COMPLETION_REASON'
    early = (count < cap if label == 'VERIFIED_EOS' and count is not None
             and type(cap) is int else False if label == 'LENGTH' else None)
    return dict(request_id=row.get('request_id'), status=row.get('status'), output_tokens=count,
        max_output_tokens=cap, finish_reason=finish, native_stop_reason_recorded='native_stop_reason' in row,
        native_stop_reason=native, final_output_token_id=final, termination=label,
        eos_evidence=evidence, verified_early_eos=early)


def output_diagnostics(raw, store, contract):
    if raw is None or not isinstance(raw.get('requests'), list):
        return dict(status='NOT_MEASURED', per_request=None, candidate_hindsight=None)
    rows = [classify_stop(row, contract) for row in raw['requests']]
    by_id = {row['request_id']: row for row in rows}
    decisions, identity = store.get('victim_decisions'), raw.get('internal_to_source')
    hindsight = dict(status='NOT_RECORDED', unique_candidates=None)
    if isinstance(decisions, list) and isinstance(identity, dict):
        observations = [row for event in decisions for row in event.get('candidates', [])]
        ids = {identity.get(row.get('request')) for row in observations}
        known_ids = ids & by_id.keys()
        early_ids = sorted(rid for rid in known_ids if by_id[rid]['verified_early_eos'] is True)
        hindsight = dict(status='OBSERVED_POSTHOC', candidate_row_observations=len(observations),
            unique_known_candidates=len(known_ids), unknown_identity_rows=sum(identity.get(row.get('request'))
                not in by_id for row in observations), verified_early_eos_requests=early_ids,
            unique_verified_early_eos=len(early_ids),
            unique_eos_or_completion_unknown=sum(by_id[rid]['verified_early_eos'] is None for rid in known_ids),
            candidate_termination_counts=dict(Counter(by_id[rid]['termination'] for rid in known_ids)),
            qualified_unique_verified_early_eos=len({identity.get(row.get('request')) for row in observations
                if row.get('qualified') is True} & set(early_ids)), semantics=SEMANTICS['hindsight'])
    return dict(status='OBSERVED', termination_counts=dict(Counter(row['termination'] for row in rows)),
        native_stop_reason_present=sum(row['native_stop_reason_recorded'] for row in rows),
        output_length_distribution=base.distribution([row['output_tokens'] for row in rows]),
        verified_early_eos_requests=[row['request_id'] for row in rows if row['verified_early_eos'] is True],
        per_request=rows, candidate_hindsight=hindsight)


def policy_diagnostics(store, raw):
    decisions = store.get('victim_decisions')
    if raw is None or not isinstance(decisions, list):
        return dict(status='NOT_RECORDED', decisions=None, pairwise_suggestion_differences=None)
    pairs = {}
    for first, second in combinations(policies.FIELDS, 2):
        names = [policies.FIELDS[key] for key in (first, second)]
        known = [event for event in decisions if all(isinstance((event.get(name) or {}).get('proposed_request'), str)
            for name in names)]
        pairs[first + '_vs_' + second] = dict(known_decisions=len(known), unknown_decisions=len(decisions)-len(known),
            different_suggestions=sum(event[names[0]]['proposed_request'] != event[names[1]]['proposed_request'] for event in known))
    return dict(status='OBSERVED', decisions=len(decisions),
        probes={rule: store.get(rule + '_probe') for rule in policies.FIELDS},
        pairwise_suggestion_differences=pairs, semantics=SEMANTICS['shadows'])


def partial_request_distributions(raw):
    """Retain observed request metrics if the inherited full-cohort validator fails."""
    if raw is None or not isinstance(raw.get('requests'), list):
        return dict(status='NOT_MEASURED', per_request=None)
    rows = []
    for request in raw['requests']:
        arrival, end = request.get('arrival_s'), request.get('completion_s')
        times, tokens = request.get('token_times_s'), request.get('output_token_ids')
        known = (base.finite(arrival) and isinstance(times, list) and isinstance(tokens, list)
            and len(times) == len(tokens) and all(base.finite(t) and t >= arrival for t in times)
            and all(a <= b for a, b in zip(times, times[1:])))
        distinct = sorted(set(times)) if known else []
        rows.append(dict(request_id=request.get('request_id'), status=request.get('status'),
            outputs=len(tokens) if isinstance(tokens, list) else None,
            ttft_s=times[0]-arrival if known and times else None,
            actual_completion_flow_s=end-arrival if request.get('status') == 'completed'
                and base.finite(arrival) and base.finite(end) and end >= arrival else None,
            max_gap_s=max((b-a for a, b in zip(distinct, distinct[1:])), default=None)))
    token_count = sum(row['outputs'] for row in rows) if all(row['outputs'] is not None for row in rows) else None
    duration = raw.get('observation_end_s')
    return dict(status='OBSERVED_PARTIAL_RECORDS_ONLY', recorded_requests=len(rows),
        distributions={key: base.distribution([row[key] for row in rows]) for key in
            ('ttft_s', 'actual_completion_flow_s', 'max_gap_s')},
        actual_observed_output_tokens=token_count, actual_observed_output_tokens_s=token_count/duration
            if token_count is not None and base.finite(duration) and duration > 0 else None,
        per_request=rows, semantics='Only recorded observed values; missing requests/timestamps are not zero or completed. No incomplete-flow penalty is presented as actual completion.')


def quality_samples(workload, raw, outputs, tokenizer_path, input_config, input_verified):
    ids = workload['task']['quality_check_ids']
    sources = {row['request_id']: row for row in workload['source_requests']}
    records = {row['request_id']: row for row in raw['requests']} if raw is not None else {}
    outcomes = {row['request_id']: row for row in outputs.get('per_request') or []}
    tokenizer, decode_status, error = None, 'NOT_REQUESTED', None
    if tokenizer_path is not None:
        decode_status = 'NOT_MEASURED' if raw is None else 'INPUT_NOT_VERIFIED'
        if raw is not None and input_verified:
            try:
                expected = input_config['source']['tokenizer_files_sha256']
                if not tokenizer_path.is_dir() or not expected or any(not (tokenizer_path / name).is_file()
                        or mixed.sha(tokenizer_path / name) != digest for name, digest in expected.items()):
                    raise ValueError('Local tokenizer files do not match the prepared Instruct input')
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
                decode_status = 'CPU_OFFLINE_INPUT_TOKENIZER_VERIFIED'
            except Exception as exc:
                decode_status, error = 'DECODE_UNAVAILABLE', f'{type(exc).__name__}: {exc}'
    samples = []
    for rid in ids:
        row, source = records.get(rid), sources[rid]
        tokens = row.get('output_token_ids') if row is not None else None
        samples.append(dict(request_id=rid, article_title=source.get('article_title'),
            article_text=source.get('article_text'), output_token_ids=tokens,
            output_text=tokenizer.decode(tokens, skip_special_tokens=False) if tokenizer is not None
                and isinstance(tokens, list) else None, outcome=outcomes.get(rid), review_status='NOT_ASSESSED'))
    return dict(status='NOT_ASSESSED', decode_status=decode_status, decode_error=error,
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        scope=workload['task']['quality_scope'], samples=samples)


def analyze(session, policy_source, input_config, tokenizer_path=None):
    plan = base.read(session / 'plan.json')
    _, spec = validate_plan(plan)
    profile = profiled.read_profile(session / 'cell-00-profile')
    check_plan = dict(plan, pinned_gpu_kv_bytes=profile['capacity'].get('pin_kv_cache_memory_bytes'))
    directory = session / 'cell-01-remaining-budget'
    previous = base.verify_choices
    try:
        base.verify_choices = policies.verify_choices
        cell, raw = base.read_native_cell(directory, spec, check_plan,
            plan.get('configuration', {}).get('max_seconds', 600), policy_source)
    finally:
        base.verify_choices = previous
    archive = base.archive_dir(directory)
    store, environment = (base.optional(archive / name, {}) for name in ('selective-store.json', 'environment.json'))
    executed = directory / 'candidate_native_oldest_strong_r01/pkg'
    source = executed / 'staged_store_rotation.py'
    source = source if source.is_file() else policy_source
    executed_config = executed / 'inputs/config.json'
    config_path = executed_config if executed_config.is_file() else input_config
    config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
    inputs = input_check(config, workload, cell, raw)
    inputs.update(path=str(config_path), config_sha256=mixed.sha(config_path),
        workload_file_sha256=mixed.sha(config_path.with_name('workload.json')))
    capacity = profiled.capacity_check(profile, cell, environment, plan)
    capacity['checks'].pop('profile_prefix_enabled')
    capacity['checks']['profile_prefix_disabled'] = (profile['config'].get('enable_prefix_caching', False) is False
        and profile['engine_args'].get('enable_prefix_caching') is False)
    capacity['status'] = 'VERIFIED' if all(capacity['checks'].values()) else 'MISMATCH_OR_MISSING'
    warmup = mixed.warmup_check(archive, cell.get('engine_args', {}))
    contract = stop_contract(cell.get('config') or {}, base.optional(archive / 'resolved-eos.json', {}),
        source.parent / 'request_measurement.py', environment.get('source_sha256', {}).get('request_measurement.py'))
    outputs = output_diagnostics(raw, store, contract)
    if isinstance(raw, dict) and isinstance(raw.get('preemption_events'), list):
        cell['native_preemptions']['unique_successful_victims'] = len(cell['native_preemptions']['per_request_counts'])
    else:
        cell['native_preemptions'] = dict(status='NOT_RECORDED', successful_calls=None,
            unique_successful_victims=None, per_request_counts=None)
    if 'metrics' in cell:
        cell['metrics'].pop('frontier', None)
        cell['metrics'].pop('frontier_role', None)
    else:
        cell['partial_request_distributions'] = partial_request_distributions(raw)
    if raw is None:
        for key in ('least_generated_shadow', 'native_selection_differences', 'policy_observations'):
            cell[key] = dict(status='NOT_MEASURED')
    choice = cell['executed_choice_check']
    checks = dict(inputs=inputs['status'] == 'VERIFIED', capacity=capacity['status'] == 'VERIFIED',
        warmup=warmup['status'] == 'VERIFIED', configuration=cell['configuration_check']['status'] == 'VERIFIED',
        choice_source=choice['source_verified'], decisions_recorded=isinstance(store.get('victim_decisions'), list),
        choices_match=all(row['status'] == 'MATCH' for row in choice['checks']),
        prefix_off=store.get('prefix_caching') is False and store.get('coordinator_type') == 'KVCacheCoordinatorNoPrefixCache')
    return dict(status='NO_MEASUREMENT' if raw is None else 'COMPLETE_FEASIBILITY_CAPTURE'
        if cell.get('status') == 'COMPLETE' and all(checks.values()) else 'INCOMPLETE_OR_INVALID',
        experiment_role=ROLE, session=str(session), plan=plan, receipt=base.optional(session / 'receipt.json'),
        semantics=SEMANTICS, checks=checks, profile_cell=profile, profile_capacity_check=capacity,
        input_consistency=inputs, fixed_warmup_check=warmup, cell=cell, stop_contract=contract,
        outputs=outputs, policy_suggestions=policy_diagnostics(store, raw),
        quality_review=quality_samples(workload, raw, outputs, tokenizer_path, config, inputs['status'] == 'VERIFIED'),
        analysis_code_sha256={Path(module.__file__).name: mixed.sha(Path(module.__file__))
            for module in (base, mixed, policies, profiled)} | {Path(__file__).name: mixed.sha(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--policy-source', type=Path, default=PACKAGE / 'staged_store_rotation.py')
    parser.add_argument('--input-config', type=Path, default=PACKAGE / 'inputs/config.json')
    parser.add_argument('--tokenizer', type=Path, help='Existing local Instruct tokenizer; CPU and local_files_only')
    parser.add_argument('--output', type=Path, help='Exclusive create; omitted means stdout only')
    args = parser.parse_args()
    result = analyze(args.session.resolve(), args.policy_source.resolve(), args.input_config.resolve(), args.tokenizer)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output))))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
