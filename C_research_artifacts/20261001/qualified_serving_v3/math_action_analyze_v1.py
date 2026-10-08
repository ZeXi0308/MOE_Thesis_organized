#!/usr/bin/env python3
"""CPU-only math episode analysis; quality is reported, never a launch gate.

Read frozen answers separately from generation artifacts. Output is exclusive.
Host-return and scheduler wrapper wall times are not device ITL or process CPU.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from health_analyze_v2 import prediction, periodic, eos_ids, native_id_mapping, token_ids
from native_pressure_analyze_v1 import analyze as check_pressure

PLANNED = 128


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def integer(value):
    return type(value) is int and value >= 0


def stats(values):
    finite = sorted(v for v in values if number(v))
    def quantile(q):
        if not finite:
            return None
        index = (len(finite) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return finite[lo] + (finite[hi] - finite[lo]) * (index - lo)
    return dict(planned=PLANNED, observed=len(finite), missing=PLANNED - len(finite),
                mean=sum(finite) / len(finite) if finite else None,
                max=max(finite) if finite else None,
                p50=quantile(.50), p90=quantile(.90), p95=quantile(.95),
                all_planned_observed=len(finite) == PLANNED)


def overlap(intervals, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def merge_interval(intervals, start, end):
    merged = []
    for a, b in sorted(intervals + [(start, end)]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def attribution(pressure, steps, events, mapping, issues):
    """Use final schedule mappings, never allocations removed by preemption."""
    calls = pressure.get('scheduler_calls', [])
    attempts = pressure.get('allocation_attempts', [])
    if not isinstance(calls, list) or not isinstance(attempts, list):
        issues.append('pressure_event_lists_invalid')
        return {}
    if not isinstance(steps, list):
        issues.append('engine_steps_not_list')
        steps = []
    if len(calls) != len(steps):
        issues.append('schedule_step_index_count_mismatch')
    by_call = defaultdict(list)
    for event in attempts:
        if isinstance(event, dict) and integer(event.get('call_id')):
            by_call[event['call_id']].append(event)
    event_tokens = Counter()
    event_requests = Counter()
    for event in events:
        if number(event.get('return_s')) and token_ids(event.get('delta_token_ids'), True):
            event_tokens[event['return_s']] += len(event['delta_token_ids'])
            event_requests[event['return_s']] += 1
    step_returns = {s.get('return_s') for s in steps if isinstance(s, dict)
                    and number(s.get('return_s'))}
    if set(event_requests) - step_returns:
        issues.append('host_events_without_matching_engine_step')
    totals, people, gates = Counter(), {}, defaultdict(Counter)
    for key in ('scheduled_tokens', 'first_prefill_tokens', 'recompute_tokens', 'decode_tokens',
                'apc_hit_tokens', 'preemption_events', 'resumed_after_preemption',
                'unattributed_scheduled_tokens', 'schedule_host_wall_s', 'engine_step_wall_s',
                'no_token_output_engine_steps', 'no_token_output_engine_step_wall_s'):
        totals[key] = 0
    executed = defaultdict(list)
    pending_preemption = Counter()
    records = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            issues.append('invalid_schedule_call:' + str(index))
            continue
        cid = call.get('call_id')
        if cid != index:
            issues.append('nonsequential_schedule_call_id:' + str(index))
        cpu = None
        if number(call.get('start_s')) and number(call.get('return_s')):
            cpu = call['return_s'] - call['start_s']
            if cpu >= 0:
                totals['schedule_host_wall_s'] += cpu
            else:
                issues.append('negative_schedule_duration:' + str(index))
                cpu = None
        else:
            issues.append('invalid_schedule_duration:' + str(index))
        step = steps[index] if index < len(steps) and isinstance(steps[index], dict) else {}
        wall = None
        if number(step.get('start_s')) and number(step.get('return_s')):
            wall = step['return_s'] - step['start_s']
            if wall < 0:
                issues.append('negative_step_duration:' + str(index))
                wall = None
            else:
                totals['engine_step_wall_s'] += wall
                if event_requests[step['return_s']] != step.get('output_requests'):
                    issues.append('step_host_output_inventory_mismatch:' + str(index))
                if not event_tokens[step['return_s']]:
                    totals['no_token_output_engine_steps'] += 1
                    totals['no_token_output_engine_step_wall_s'] += wall
        else:
            issues.append('invalid_step_duration:' + str(index))
        for label in set(call.get('classifications', [])):
            gates[label]['calls'] += 1
            if wall is not None:
                gates[label]['engine_step_wall_s'] += wall
            if cpu is not None:
                gates[label]['schedule_host_wall_s'] += cpu
        for rid in call.get('preempted_request_ids', []):
            pending_preemption[rid] += 1
            totals['preemption_events'] += 1
            if rid not in mapping:
                issues.append('unknown_preempted_native_id:' + rid)
        scheduled = call.get('scheduled_tokens', {})
        if not isinstance(scheduled, dict):
            issues.append('invalid_scheduled_mapping:' + str(index))
            continue
        for rid, count in scheduled.items():
            if rid not in mapping or not integer(count):
                issues.append('unknown_or_invalid_scheduled_request:' + str(rid))
                continue
            totals['scheduled_tokens'] += count
            person = people.setdefault(mapping[rid], Counter())
            person['scheduled_tokens'] += count
            success = [e for e in by_call[cid] if e.get('request_id') == rid
                       and e.get('succeeded') is True and e.get('returned_none') is False]
            if not success:
                issues.append('scheduled_without_successful_allocation:' + str(cid) + ':' + rid)
                totals['unattributed_scheduled_tokens'] += count
                continue
            event = success[-1]
            args = event.get('arguments', {})
            fields = [event.get(k) for k in ('num_prompt_tokens', 'num_tokens', 'num_computed_tokens')]
            hit = args.get('num_new_computed_tokens')
            if not all(integer(v) for v in fields + [hit]) or fields[1] < fields[0]:
                issues.append('invalid_allocation_token_geometry:' + str(cid) + ':' + rid)
                totals['unattributed_scheduled_tokens'] += count
                continue
            prompt, history, computed = fields
            outputs = history - prompt
            start, end = computed + hit, computed + hit + count
            # The latest generated token at history-1 is ordinary decode input.
            # Positions after that are also not recomputation of past work.
            if outputs:
                recompute = max(0, min(end, history - 1) - start)
                first = 0
            else:
                prompt_end = min(end, prompt)
                recompute = overlap(executed[rid], start, prompt_end)
                first = max(0, prompt_end - start) - recompute
            decode = count - recompute - first
            resumed = bool(pending_preemption[rid])
            if resumed:
                pending_preemption[rid] = 0
            values = dict(first_prefill_tokens=first, recompute_tokens=recompute,
                          decode_tokens=decode, apc_hit_tokens=hit,
                          resumed_after_preemption=int(resumed))
            for key, value in values.items():
                totals[key] += value
                person[key] += value
            executed[rid] = merge_interval(executed[rid], start, end)
            records.append(dict(call_id=cid, request_id=mapping[rid], native_request_id=rid,
                                scheduled_tokens=count, output_history_tokens=outputs,
                                computed_before=computed, new_computed_hit=hit,
                                span_start=start, span_end=end, **values))
    totals['unrecovered_preemption_events'] = sum(pending_preemption.values())
    return dict(totals=dict(totals), per_request={k: dict(v) for k, v in people.items()},
                scheduled_work=records, gate_time_coverage={k: dict(v) for k, v in gates.items()},
                definition='Call i is paired with engine step i. Only final scheduled_tokens '
                'use the last successful allocation for that request/call. For O>0, recompute '
                'is the scheduled span intersected with [0, num_tokens-1); normal decode is excluded. '
                'For O=0, previously executed prompt positions are recompute, new positions are first prefill.',
                limitations=['Gate coverage may overlap; it is not an additive causal decomposition.',
                             'Scheduler duration is instrumented host wall time, not measured process CPU.',
                             'No full per-request queue snapshots exist; exact waiting-time decomposition is unavailable.',
                             'APC hits are not newly executed prefill tokens. Schedule work counts are tokens, not GPU time.'])


def analyze(run, answers_path):
    run, answers_path = Path(run), Path(answers_path)
    answers = json.loads(answers_path.read_text())
    if not isinstance(answers, dict) or len(answers) != PLANNED or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in answers.items()):
        raise ValueError('answers must contain exactly 128 request_id -> gold string entries')
    issues, hashes = [], {'answers': sha(answers_path)}

    def read(name, default):
        path = run / name
        try:
            hashes[name] = sha(path)
            result = json.loads(path.read_text())
            if not isinstance(result, type(default)):
                issues.append(name + ':unexpected_json_type')
                return default
            return result
        except (OSError, ValueError, UnicodeError) as exc:
            issues.append(name + ':' + type(exc).__name__)
            return default

    source = read('source-input.json', {})
    rendered = read('rendered-inputs.json', [])
    outputs = read('measured-outputs.json', [])
    status = read('status.json', {})
    config = read('config.json', {})
    tokenizer = read('tokenizer.json', {})
    eos = eos_ids(read('resolved-eos.json', {}))
    sampling = read('measured-native-sampling.json', {})
    reset = read('prefix-cache-reset.json', {})
    drain = read('native-drain.json', {})
    provenance = read('provenance.json', {})
    pressure = read('measured-pressure.json', {})
    runtime = read('resolved-runtime.json', {})
    steps_receipt = read('measured-steps.json', {})
    if status.get('status') != 'COMPLETE' or status.get('expected_requests') != PLANNED:
        issues.append('cell_incomplete')
    if source.get('arrival_traces_s') != [0.0] * PLANNED:
        issues.append('source_arrivals_not_zero_burst')
    frozen_path = answers_path.parent / 'math_inputs128_v1.json'
    try:
        hashes['frozen_inputs'] = sha(frozen_path)
        if source != json.loads(frozen_path.read_text()) or provenance.get('input_sha256') != hashes['frozen_inputs']:
            issues.append('source_or_provenance_frozen_input_mismatch')
    except (OSError, ValueError, UnicodeError) as exc:
        issues.append('frozen_inputs:' + type(exc).__name__)
    if provenance.get('answer_keys_loaded') is not False:
        issues.append('answer_key_isolation_not_attested')
    if not tokenizer.get('chat_template') or tokenizer.get('answer_cue') != 'Answer:':
        issues.append('tokenizer_template_receipt_invalid')
    if not eos:
        issues.append('missing_model_eos_ids')
    args, sample = config.get('engine_args', {}), config.get('sampling', {})
    cap = sample.get('max_tokens')
    if not integer(cap) or cap < 1:
        issues.append('invalid_output_cap')
        cap = 1024
    if not (args.get('enable_prefix_caching') is True and args.get('async_scheduling') is False):
        issues.append('apc_sync_contract_mismatch')
    if cap != 1024 or args.get('max_model_len') != 4096:
        issues.append('math_context_or_cap_contract_mismatch')
    for resolved, configured in (('max_num_running_reqs', 'max_num_seqs'),
                                 ('max_num_scheduled_tokens', 'max_num_batched_tokens'),
                                 ('kv_cache_memory_bytes', 'kv_cache_memory_bytes'),
                                 ('scheduler_reserve_full_isl', 'scheduler_reserve_full_isl')):
        if runtime.get(resolved) != args.get(configured):
            issues.append('resolved_config_mismatch:' + resolved)
    if any(sample.get(k) != v for k, v in dict(temperature=0.0, min_tokens=0,
                                               ignore_eos=False, stop=[]).items()):
        issues.append('sampling_contract_mismatch')
    if not (reset.get('reset_succeeded') is True and reset.get('cached_hash_keys_after') == 0
            and reset.get('before', {}).get('status') == reset.get('after', {}).get('status') == 'QUALIFIED'):
        issues.append('cold_apc_reset_unqualified')
    if not (drain.get('status') == 'QUALIFIED' and drain.get('unfinished') is False
            and drain.get('requests') == drain.get('running') == drain.get('waiting') == 0
            and integer(drain.get('total_blocks')) and drain.get('total_blocks', 0) > 1
            and drain.get('free_blocks') == drain['total_blocks'] - 1):
        issues.append('native_drain_unqualified')
    native, identity_errors = native_id_mapping(sampling, {'measured/' + rid for rid in answers})
    issues.extend(identity_errors)
    bad_sampling = set()
    for rid, params in (sampling.items() if isinstance(sampling, dict) else []):
        params = params if isinstance(params, dict) else {}
        stop_ids, all_stop = params.get('stop_token_ids') or [], params.get('all_stop_token_ids') or []
        if not (params.get('stop') in (None, []) and params.get('ignore_eos') is False
                and params.get('min_tokens') == 0 and params.get('max_tokens') == cap
                and token_ids(stop_ids, True) and token_ids(all_stop, True)
                and set(stop_ids + all_stop).issubset(eos)
                and (params.get('_eos_token_id') in eos or bool(set(all_stop) & set(eos)))):
            issues.append('native_eos_only_sampling_invalid:' + str(rid))
            bad_sampling.add(native.get(rid))

    def inventory(rows, label):
        result = {}
        for row in rows if isinstance(rows, list) else []:
            rid = row.get('request_id') if isinstance(row, dict) else None
            if rid not in answers or rid in result:
                issues.append(label + ':unknown_or_duplicate_id:' + str(rid))
            elif isinstance(row, dict):
                result[rid] = row
        if set(result) != set(answers):
            issues.append(label + ':inventory_mismatch')
        return result

    sources = inventory(source.get('requests', []), 'source')
    renders = inventory(rendered, 'rendered')
    raws = inventory(outputs, 'outputs')
    rows, all_events = [], []
    for rid, gold in answers.items():
        src, render, row = sources.get(rid, {}), renders.get(rid, {}), raws.get(rid, {})
        errors = []
        if not src or not render or not row:
            errors.append('missing_source_rendered_or_output')
        if any(k in src for k in ('gold', 'answer', 'answer_key')):
            errors.append('answer_key_in_generation_source')
        if any(render.get(k) != v or row.get(k) != v for k, v in src.items()):
            errors.append('source_rendered_output_mismatch')
        prompt_ids = render.get('prompt_token_ids')
        if not token_ids(prompt_ids) or row.get('prompt_token_ids') != prompt_ids:
            errors.append('prompt_ids_mismatch_or_invalid')
        if render.get('prompt') != row.get('prompt') or not isinstance(render.get('prompt'), str):
            errors.append('prompt_text_mismatch_or_invalid')
        if token_ids(prompt_ids):
            digest = hashlib.sha256(json.dumps(prompt_ids, separators=(',', ':')).encode()).hexdigest()
            if render.get('prompt_token_ids_sha256') != digest or render.get('prompt_tokens') != len(prompt_ids):
                errors.append('prompt_token_geometry_or_hash_mismatch')
            if len(prompt_ids) + cap > 4096 or not str(render.get('prompt', '')).endswith('Answer:'):
                errors.append('prompt_context_or_answer_cue_invalid')
        if row.get('external_request_id') != 'measured/' + rid or row.get('arrival_s') != 0.0:
            errors.append('external_identity_or_arrival_mismatch')
        ids, text = row.get('output_token_ids'), row.get('output_text')
        if not token_ids(ids, True) or len(ids) > cap:
            errors.append('invalid_output_tokens')
            ids = ids if token_ids(ids, True) else []
        if not isinstance(text, str):
            errors.append('invalid_output_text')
            text = ''
        events = row.get('host_returns', [])
        if not isinstance(events, list):
            errors.append('invalid_host_returns')
            events = []
        cumulative, times, valid_events, finished_seen, previous = [], [], [], False, -1.0
        for event in events:
            if not (isinstance(event, dict) and event.get('request_id') == 'measured/' + rid
                    and number(event.get('return_s')) and event['return_s'] >= previous
                    and token_ids(event.get('delta_token_ids'), True) and not finished_seen):
                errors.append('invalid_host_event')
                continue
            previous = event['return_s']
            cumulative.extend(event['delta_token_ids'])
            times.extend([previous] * len(event['delta_token_ids']))
            if event.get('cumulative_tokens') != len(cumulative):
                errors.append('host_event_token_count_mismatch')
            finished_seen = event.get('finished') is True
            valid_events.append(event)
        all_events.extend(valid_events)
        if cumulative != ids or row.get('token_times_s') != times:
            errors.append('host_event_output_mismatch')
        completed = row.get('finished') is True and row.get('finish_reason') in ('stop', 'length')
        completion = row.get('host_elapsed_s') if completed else None
        if completed and not (valid_events and finished_seen and number(completion)
                              and completion == valid_events[-1]['return_s']
                              and valid_events[-1].get('finish_reason') == row.get('finish_reason')
                              and valid_events[-1].get('stop_reason') == row.get('stop_reason')):
            errors.append('completion_event_mismatch')
            completion = None
        if row.get('finish_reason') == 'length' and len(ids) != cap:
            errors.append('length_finish_without_cap')
        stop = row.get('stop_reason')
        natural = completed and row.get('finish_reason') == 'stop' and bool(eos) and (
            stop is None or type(stop) is int and stop in eos) and 'measured/' + rid not in bad_sampling
        if row.get('finish_reason') == 'stop' and not natural:
            errors.append('unexplained_non_eos_stop')
        ttft = times[0] if times else None
        gap = max((b - a for a, b in zip(times, times[1:])), default=0.0) if times else None
        pred = prediction(text)
        correct = completed and not errors and pred == gold
        slo = completed and not errors and ttft is not None and gap is not None and ttft <= 20.0 and gap <= 4.0
        rows.append(dict(request_id=rid, gold=gold, prediction=pred, completed=completed,
                         correct=correct, natural_eos=natural, empty=not text.strip(),
                         truncated=row.get('finish_reason') == 'length', output_tokens=len(ids),
                         finish_reason=row.get('finish_reason'), stop_reason=stop,
                         periodic_suffix_period=periodic(ids), ttft_s=ttft, completion_s=completion,
                         max_host_gap_s=gap, slo_pass=slo, correct_and_slo=correct and slo, issues=errors))
        issues.extend(rid + ':' + error for error in errors)
    try:
        journal_path = run / 'measured-host-returns.jsonl'
        hashes[journal_path.name] = sha(journal_path)
        journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
        canonical = lambda e: json.dumps(e, sort_keys=True, separators=(',', ':'))
        if Counter(map(canonical, journal)) != Counter(map(canonical, all_events)):
            issues.append('host_journal_output_inventory_mismatch')
    except (OSError, ValueError, UnicodeError) as exc:
        issues.append('host_journal:' + type(exc).__name__)
    pressure_summary = None
    try:
        pressure_summary = check_pressure(pressure, runtime)
        if pressure.get('status') != 'COMPLETE':
            issues.append('pressure_incomplete')
        if any(pressure_summary['counts'][k] for k in
               ('allocation_calls_outside_schedule', 'allocation_exceptions', 'schedule_exceptions')):
            issues.append('pressure_has_outside_calls_or_exceptions')
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        issues.append('pressure_structure_invalid:' + str(exc))
    work = attribution(pressure, steps_receipt.get('steps', []), all_events, native, issues)
    calls = pressure.get('scheduler_calls', [])
    calls = calls if isinstance(calls, list) else []
    preempted = [rid for c in calls if isinstance(c, dict) for rid in c.get('preempted_request_ids', [])]
    if preempted != steps_receipt.get('preempted_request_ids') or len(preempted) != status.get('preemptions'):
        issues.append('preemption_receipts_mismatch')
    duration = status.get('observation_end_s')
    latest = max((e['return_s'] for e in all_events), default=0)
    if not number(duration) or duration <= 0 or duration < latest:
        issues.append('invalid_episode_duration')
        duration = None
    counts = {key: sum(r[key] for r in rows) for key in
              ('completed', 'correct', 'natural_eos', 'empty', 'truncated', 'slo_pass', 'correct_and_slo')}
    counts.update(planned=PLANNED, periodic_suffix=sum(r['periodic_suffix_period'] is not None for r in rows),
                  output_tokens=sum(r['output_tokens'] for r in rows))
    if status.get('output_tokens') != counts['output_tokens'] or status.get('request_count') != PLANNED:
        issues.append('status_output_aggregate_mismatch')
    rate = lambda n: n / duration if duration else None
    return dict(schema='c-math-action-analysis-v1', integrity='VALID' if not issues else 'INVALID',
                run=str(run.resolve()), input_sha256=hashes, analyzer_sha256=sha(__file__),
                issues=issues, counts=counts, accuracy=counts['correct'] / PLANNED,
                eos_rate=counts['natural_eos'] / PLANNED, quality_gate_applied=False,
                slo_pass_rate=counts['slo_pass'] / PLANNED,
                correct_and_slo_pass_rate=counts['correct_and_slo'] / PLANNED,
                timing={key: stats([r[key] for r in rows]) for key in
                        ('ttft_s', 'completion_s', 'max_host_gap_s', 'output_tokens')},
                episode=dict(duration_s=duration, planned_requests_per_s=rate(PLANNED),
                             completed_requests_per_s=rate(counts['completed']),
                             output_tokens_per_s=rate(counts['output_tokens']),
                             slo_goodput_requests_per_s=rate(counts['slo_pass']),
                             correct_and_slo_goodput_requests_per_s=rate(counts['correct_and_slo'])),
                diagnostic_thresholds=dict(ttft_s=20.0, max_host_gap_s=4.0,
                    gap_definition='Maximum successive emitted-token host-return gap; excludes arrival-to-first-token.',
                    scope='Historical diagnostic only, not a business or final-paper SLO. The 4s limit may not bind.'),
                per_request=rows, attribution=work,
                pressure_counts=pressure_summary['counts'] if pressure_summary else None,
                limitations=['All 128 planned requests remain the accuracy and success-rate denominator; errors and truncations remain.',
                             'Observed timing summaries include every available request, regardless of correctness/EOS/truncation; missing counts are explicit.',
                             'Incomplete or invalid runs retain diagnostic numbers, which are not validated performance results.',
                             'Finite-burst, instrumented host timing; no steady-state, per-token device ITL, or exact causal waiting decomposition.',
                             'Correctness uses the inherited last-number string comparison, not a new semantic grader.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--answers', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.answers)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('integrity', 'counts', 'episode')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
