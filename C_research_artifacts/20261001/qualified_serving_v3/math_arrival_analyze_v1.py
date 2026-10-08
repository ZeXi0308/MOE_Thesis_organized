#!/usr/bin/env python3
"""Frozen two-burst math analysis. Reuse scale grading and unchanged raw-clock attribution.

The analysis body derives from the pinned scale analyzer with explicit arrival
validation and request-latency changes; its original file is never modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import math_scale_analyze_v1 as base
from math_scale_analyze_v1 import (PLANNED, sha, number, integer, stats, attribution,
    prediction, periodic, eos_ids, native_id_mapping, token_ids, check_pressure)

BASE_ANALYZER_SHA = '7dd48c582ae0672b8bdc2b1cf8db8cb626ec5740f9207e2aa81c5ff1870411f4'
ORIGINAL_INPUT_SHA = '513ae9ac6dff4e36693c07c1c37cd9d6d6101612beda5aad59bb33994104e13a'
FROZEN_INPUT_SHA = '29c9a46a8076cc10f78bd6cecbe7e4fc41a4e29e2299f901128a91e52ca359d1'
TIMING_KEYS = ('ttft_s', 'completion_s', 'completion_at_s', 'max_host_gap_s', 'output_tokens',
               'submission_lag_s', 'submission_return_lag_s', 'add_request_wall_s')
METRICS = dict(ttft_s='First emitted-token host return minus offered arrival_s.',
    completion_s='Flow: final host_elapsed_s minus offered arrival_s; includes submission lag.',
    completion_at_s='Final host_elapsed_s on unchanged episode clock.',
    submission_lag_s='Actual add_request start minus offered arrival_s.',
    submission_return_lag_s='Actual add_request return minus offered arrival_s.',
    add_request_wall_s='Actual add_request return minus start.',
    max_host_gap_s='Maximum successive emitted-token host-return gap, unchanged; excludes arrival-to-first-token.',
    attribution='Original episode-relative raw host events and engine steps; schedule i joins engine step i.')


def cohort_summary(rows, arrival):
    group = [r for r in rows if r['arrival_s'] == arrival]
    counts = {k:sum(r[k] for r in group) for k in
        ('completed', 'correct', 'natural_eos', 'empty', 'truncated', 'slo_pass', 'correct_and_slo', 'output_tokens')}
    counts['planned'] = 512
    timing = {key:stats([r[key] for r in group]) for key in TIMING_KEYS}
    for summary in timing.values():
        summary.update(planned=512, missing=512-summary['observed'], all_planned_observed=summary['observed']==512)
    return dict(arrival_s=arrival, counts=counts, timing=timing, accuracy=counts['correct']/512,
        eos_rate=counts['natural_eos']/512, slo_pass_rate=counts['slo_pass']/512,
        correct_and_slo_pass_rate=counts['correct_and_slo']/512)


def validate_arrival_receipt(receipt, offered, rows, mapping, steps, issues):
    """Validate submission and native-ID scans without shifting any raw clocks."""
    def check(ok, error):
        if not ok: issues.append(error)
    if receipt.get('schema') != 'c-arrival-generate-v1' or receipt.get('offered_arrivals_s') != offered:
        issues.append('arrival_receipt_schema_or_offered_mismatch')
    check(receipt.get('status') == 'COMPLETE' and receipt.get('hooks_restored') is True,
          'arrival_generator_incomplete_or_hook_not_restored')
    for field in ('planned_requests', 'submitted_requests', 'finished_requests', 'native_sampling_count'):
        check(receipt.get(field) == PLANNED, 'arrival_receipt_count:' + field)
    clock = receipt.get('clock', {})
    check(number(clock.get('perf_counter_origin_s')) and number(clock.get('epoch_origin_unix_s')), 'arrival_clock_missing')
    for rid, row in rows.items():
        expected = clock.get('epoch_origin_unix_s', 0) + row.get('arrival_s', 0)
        check(row.get('native_arrival_time_unix_s') == expected, 'native_offered_arrival_mismatch:' + rid)
    cohorts = receipt.get('cohorts', [])
    check(len(cohorts) == 2 and [c.get('arrival_s') for c in cohorts] == [0.0, 4.0], 'arrival_cohort_inventory')
    for cohort in cohorts:
        arrival = cohort.get('arrival_s')
        group = [r for r in rows.values() if r.get('arrival_s') == arrival]
        submitted = [r for r in group if r.get('submitted') is True and number(r.get('add_request_s'))
                     and number(r.get('add_request_return_s'))]
        check(cohort.get('planned_requests') == len(group) == 512 and cohort.get('submitted_requests') == len(submitted)
              and cohort.get('finished_requests') == sum(r.get('finished') is True for r in group), 'arrival_cohort_counts')
        check(cohort.get('min_add_request_s') == min((r['add_request_s'] for r in submitted), default=None)
              and cohort.get('max_add_request_return_s') == max((r['add_request_return_s'] for r in submitted), default=None)
              and cohort.get('max_submission_lag_s') == max((r['add_request_s']-arrival for r in submitted), default=None),
              'arrival_cohort_submission_times')
    scans, engine_steps = receipt.get('sampling_scans', []), steps.get('steps', [])
    check(isinstance(scans, list) and bool(scans), 'native_sampling_scans_missing')
    if not isinstance(scans, list) or not scans: return
    initial, seen, previous = scans[0].get('before_scheduler_step'), [], -1
    check(integer(initial), 'sampling_scan_initial_step_invalid')
    for scan in scans:
        before, ids = scan.get('before_scheduler_step'), scan.get('new_native_request_ids', [])
        if not integer(initial) or not integer(before) or not isinstance(ids, list):
            check(False, 'sampling_scan_shape_invalid'); continue
        index = before - initial
        check(index > previous and 0 <= index < len(engine_steps), 'sampling_scan_step_order')
        previous = index; seen.extend(ids)
        check(bool(ids) and scan.get('native_sampling_count') == len(seen) == scan.get('submitted_requests'),
              'sampling_scan_cumulative_count')
        if not 0 <= index < len(engine_steps): continue
        start = engine_steps[index].get('start_s')
        if not number(start): check(False, 'sampling_scan_step_start_invalid'); continue
        check(scan.get('submitted_requests') == sum(number(r.get('add_request_return_s'))
              and r['add_request_return_s'] <= start for r in rows.values()), 'sampling_scan_submission_clock')
        for rid in ids:
            source_id = mapping.get(rid, '')[len('measured/'):]
            row = rows.get(source_id, {})
            check(rid in mapping and number(row.get('add_request_return_s')) and row['add_request_return_s'] <= start,
                  'sampling_scan_before_submission:' + str(rid))
    check(Counter(seen) == Counter(mapping.keys()), 'sampling_scan_native_id_inventory')


def analyze(run, answers_path):
    run, answers_path = Path(run), Path(answers_path)
    if sha(base.__file__) != BASE_ANALYZER_SHA: raise ValueError('frozen base analyzer changed')
    answers = json.loads(answers_path.read_text())
    if not isinstance(answers, dict) or len(answers) != PLANNED or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in answers.items()):
        raise ValueError('answers must contain exactly 1024 request_id -> gold string entries')
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
    eos_receipt = read('resolved-eos.json', {})
    eos = eos_ids(eos_receipt)
    repetition_penalty = eos_receipt.get('sampling_defaults', {}).get('repetition_penalty', 1.0)
    sampling = read('measured-native-sampling.json', {})
    reset = read('prefix-cache-reset.json', {})
    drain = read('native-drain.json', {})
    provenance = read('provenance.json', {})
    pressure = read('measured-pressure.json', {})
    runtime = read('resolved-runtime.json', {})
    steps_receipt = read('measured-steps.json', {})
    if status.get('status') != 'COMPLETE' or status.get('expected_requests') != PLANNED:
        issues.append('cell_incomplete')
    offered = [0.0] * 512 + [4.0] * 512
    if source.get('arrival_traces_s') != offered:
        issues.append('source_arrivals_not_frozen_two_burst')
    arrivals = {r['request_id']: t for r, t in zip(source.get('requests', []), offered)}
    arrival_receipt = read('measured-arrival-receipt.json', {})
    frozen_path = answers_path.parent / 'math_twoburst_inputs1024_v1.json'
    try:
        hashes['frozen_inputs'] = sha(frozen_path)
        original_path = answers_path.parent / 'math_inputs1024_v1.json'
        hashes['original_zero_arrival_inputs'] = sha(original_path)
        original = json.loads(original_path.read_text())
        if hashes['frozen_inputs'] != FROZEN_INPUT_SHA or hashes['original_zero_arrival_inputs'] != ORIGINAL_INPUT_SHA:
            issues.append('frozen_input_sha_mismatch')
        if original.get('arrival_traces_s') != [0.0] * PLANNED or {k:v for k,v in source.items() if k != 'arrival_traces_s'} != {k:v for k,v in original.items() if k != 'arrival_traces_s'}:
            issues.append('input_changed_beyond_arrivals')
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
                and params.get('temperature') == 0.0 and params.get('repetition_penalty') == repetition_penalty
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
        arrival = arrivals.get(rid, 0.0)
        added, submitted = row.get('add_request_s'), row.get('add_request_return_s')
        valid_submit = number(added) and number(submitted) and arrival <= added <= submitted
        if row.get('submitted') is not True or not valid_submit:
            errors.append('invalid_actual_submission_receipt')
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
        if row.get('external_request_id') != 'measured/' + rid or row.get('arrival_s') != arrival or not number(row.get('arrival_s')):
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
        ttft = times[0] - arrival if times else None
        flow = completion - arrival if completion is not None else None
        if (ttft is not None and not number(ttft)) or (flow is not None and not number(flow)):
            errors.append('negative_arrival_relative_latency')
        if valid_submit and valid_events and valid_events[0]['return_s'] < submitted:
            errors.append('host_output_precedes_submission_return')
        gap = max((b - a for a, b in zip(times, times[1:])), default=0.0) if times else None
        pred = prediction(text)
        correct = completed and not errors and pred == gold
        slo = completed and not errors and ttft is not None and gap is not None and ttft <= 20.0 and gap <= 4.0
        rows.append(dict(request_id=rid, gold=gold, prediction=pred, completed=completed,
                         correct=correct, natural_eos=natural, empty=not text.strip(),
                         truncated=row.get('finish_reason') == 'length', output_tokens=len(ids),
                         finish_reason=row.get('finish_reason'), stop_reason=stop,
                         periodic_suffix_period=periodic(ids), ttft_s=ttft, completion_s=flow,
                         completion_at_s=completion, first_token_at_s=times[0] if times else None, arrival_s=arrival,
                         add_request_s=added, add_request_return_s=submitted,
                         submission_lag_s=added-arrival if valid_submit else None,
                         submission_return_lag_s=submitted-arrival if valid_submit else None,
                         add_request_wall_s=submitted-added if valid_submit else None,
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
    validate_arrival_receipt(arrival_receipt, offered, raws, native, steps_receipt, issues)
    rate = lambda n: n / duration if duration else None
    return dict(schema='c-math-arrival-analysis-v1', integrity='VALID' if not issues else 'INVALID',
                run=str(run.resolve()), input_sha256=hashes, analyzer_sha256=sha(__file__),
                issues=issues, counts=counts, base_analyzer_sha256=BASE_ANALYZER_SHA,
                arrival_contract=dict(offered_arrivals_s=offered, cohorts=2, cohort_size=512,
                    only_source_change='arrival_traces_s', original_input_sha256=ORIGINAL_INPUT_SHA,
                    frozen_input_sha256=FROZEN_INPUT_SHA), metrics_definitions=METRICS,
                cohorts=[cohort_summary(rows, t) for t in (0.0, 4.0)], accuracy=counts['correct'] / PLANNED,
                eos_rate=counts['natural_eos'] / PLANNED, quality_gate_applied=False,
                slo_pass_rate=counts['slo_pass'] / PLANNED,
                correct_and_slo_pass_rate=counts['correct_and_slo'] / PLANNED,
                timing={key: stats([r[key] for r in rows]) for key in
                        TIMING_KEYS},
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
                limitations=['All 1024 planned requests remain the accuracy and success-rate denominator; errors and truncations remain.',
                             'Observed timing summaries include every available request, regardless of correctness/EOS/truncation; missing counts are explicit.',
                             'Incomplete or invalid runs retain diagnostic numbers, which are not validated performance results.',
                             'Two finite offered bursts, instrumented host timing; submission lag remains inside offered-arrival TTFT/flow.',
                             'Raw host journal, token times, engine steps and pressure attribution retain episode clocks; none are shifted per request.',
                             'Compare policies only within this frozen arrival scenario; completion_s is flow, completion_at_s is absolute episode time.',
                             'Correctness uses the inherited last-number string comparison, not a new semantic grader.'])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'answers', 'output'): parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.answers)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('integrity', 'counts', 'episode')}, ensure_ascii=False))
    sys.exit(0 if result['integrity'] == 'VALID' else 2)


if __name__ == '__main__':
    main()
