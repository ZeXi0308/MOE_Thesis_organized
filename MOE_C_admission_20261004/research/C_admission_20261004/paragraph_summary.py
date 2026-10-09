#!/usr/bin/env python3
"""One newline-terminated baseline: population, visible text and observed gates.

Usage: python3 paragraph_summary.py CELL --output NEW.json
No policy comparison or task-quality score is inferred.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from analyze import distribution, finite, request_rows


SAMPLE_INDICES = (0, 1, 64, 128, 256, 383)


def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def extrema(records):
    result = {}
    for key in ('active', 'free_blocks', 'running'):
        values = [r.get(key) for r in records]
        if any(v is not None and not finite(v) for v in values):
            raise ValueError('Invalid observed state: '+key)
        known = [v for v in values if v is not None]
        result[key] = dict(observed_n=len(known), missing_n=len(values)-len(known),
            min=min(known) if known else None, max=max(known) if known else None)
    return result


def counts(records):
    return dict(evaluations=len(records),
        unique_requests=len({r['request_id'] for r in records}))


def summarize(cell):
    raw, raw_sha = read(cell/'raw.json')
    admission, admission_sha = read(cell/'admission.json')
    config, config_sha = read(cell/'config.json')
    stops = config['stop_strings']
    if stops not in (['\n\n'], ['\n']) or config['ignore_eos'] is not False:
        raise ValueError('Expected the declared single/double-newline natural-EOS task')
    stop_string = stops[0]
    stop_label = 'single_newline' if stop_string == '\n' else 'double_newline'
    buffer_characters = len(stop_string)-1
    requests = raw['requests']
    if len(requests) != 384 or config['requests'] != len(requests):
        raise ValueError('Expected all 384 source requests in source order')
    metrics = request_rows(raw)
    n = len(requests)
    termination = dict(double_newline=0, other_stop=0, length=0, other_completed=0, not_completed=0)
    if stop_label == 'single_newline':
        termination[stop_label] = 0  # Keep the legacy double_newline field literal, not an alias.
    native_reasons = collections.Counter()
    texts, visible_delays, visible_after_token = [], [], []
    for req, metric in zip(requests, metrics):
        # These keys distinguish an unobserved value (None) from a missing recorder.
        text, visible, native_reason = (req[k] for k in
            ('output_text', 'first_visible_text_s', 'native_stop_reason'))
        if text is not None and not isinstance(text, str):
            raise ValueError('Invalid final text: '+req['request_id'])
        if visible is not None:
            if (not finite(visible) or not req['arrival_s'] <= visible <= raw['observation_end_s']+1e-6
                    or (metric['outcome'] == 'completed' and visible > req['completion_s'])):
                raise ValueError('Invalid first visible text time: '+req['request_id'])
            visible_delays.append(visible-req['arrival_s'])
            if metric['first_token_s'] is not None:
                visible_after_token.append(visible-metric['first_token_s'])
        if metric['outcome'] != 'completed':
            termination['not_completed'] += 1
            continue
        if text is None:
            raise ValueError('Completed newline-terminated request has no recorded final text')
        texts.append(text)
        reason = req['stop_reason']  # This recorder field stores finish_reason.
        label = (stop_label if reason == 'stop' and native_reason == stop_string else
                 'other_stop' if reason == 'stop' else 'length' if reason == 'length' else
                 'other_completed')
        termination[label] += 1
        native_reasons[json.dumps(native_reason, ensure_ascii=False)] += 1

    decisions, snapshots = admission['decisions'], admission['snapshots']
    for row in decisions:
        if row['native_fit'] is not None and type(row['native_fit']) is not bool:
            raise ValueError('Invalid native_fit value')
        if type(row['denied']) is not bool:
            raise ValueError('Missing/invalid gate denial flag')
    denied = [r for r in decisions if r['denied']]
    allocations = admission['native_allocations']
    if any(type(r['actual']) is not bool for r in allocations):
        raise ValueError('Missing/invalid actual native allocation result')
    preemptions = raw.get('actual_preemption_count')
    if preemptions is not None and (type(preemptions) is not int or preemptions < 0):
        raise ValueError('Invalid actual preemption count')
    preemption_events = raw.get('preemption_events')
    if preemption_events is not None:
        if any(type(e['original_preemption_returned']) is not bool for e in preemption_events):
            raise ValueError('Missing/invalid preemption method-return flag')
        observed = sum(e['original_preemption_returned'] for e in preemption_events)
        if preemptions is not None and preemptions != observed:
            raise ValueError('Preemption count disagrees with recorded events')
        preemptions = observed

    return dict(schema='paragraph-baseline-summary-v1', cell=str(cell),
        sources_sha256=dict(raw=raw_sha, admission=admission_sha, config=config_sha),
        configuration=config, raw_status=raw['status'], raw_error=raw.get('error'),
        task_termination=dict(task='wikitext_source_line_continuation' if stop_label == 'single_newline'
            else 'generated_double_newline_continuation', stop_strings=stops,
            configured_stop_label=stop_label, include_stop_str_in_output=False,
            streaming_buffer_characters=buffer_characters,
            semantics='A source line may be prose or a heading. Neither stop rule guarantees '
                'a complete natural-language paragraph; EOS or the token limit may terminate first.'),
        planned_requests=n, observation_end_s=raw['observation_end_s'],
        outcomes=dict(collections.Counter(r['outcome'] for r in metrics)),
        not_arrived_at_observation_end=sum(not r['arrived_at_observation_end'] for r in metrics),
        termination=termination, completed_native_stop_reasons_json_keys=dict(native_reasons),
        final_visible_text=dict(observed_n=len(texts), missing_n=n-len(texts),
            empty=sum(t == '' for t in texts),
            whitespace_only_nonempty=sum(bool(t) and not t.strip() for t in texts),
            contains_nonwhitespace=sum(bool(t.strip()) for t in texts),
            semantics='Final text is recorded only on completed requests. Missing is not empty. '
                'First visible text means nonempty text, including whitespace.'),
        token_ttft_s=distribution([r['ttft_s'] for r in metrics], n),
        first_visible_text_latency_s=distribution(visible_delays, n),
        first_visible_minus_first_token_s=distribution(visible_after_token, n),
        completion_latency_s=distribution([r['flow_s'] for r in metrics], n),
        output_tokens=distribution([r['output_tokens'] for r in metrics], n),
        timing_semantics='All latencies start at external arrival. Distributions retain all planned '
            'requests in their CDF denominator; quantiles condition on observed values. Token IDs '
            'include generated stop tokens; visible text excludes the configured stop string and '
            f'buffers {buffer_characters} trailing characters while unfinished. '
            'Token TTFT is not first visible text. Missing text time remains missing. '
            'Host engine.step return includes detokenization and stop handling costs.',
        observed_state=dict(snapshot_count=len(snapshots), gate_decision_count=len(decisions),
            snapshots=extrema(snapshots), gate_decisions=extrema(decisions),
            combined=extrema(snapshots+decisions),
            semantics='Extrema of recorded snapshots and actual gate evaluations only; not continuous '
                'GPU peaks. Missing fields remain missing; gate sampling is decision-dependent.'),
        admission_observations=dict(
            new_gate_native_fit_refusals=counts([r for r in decisions if r['native_fit'] is False]),
            new_gate_native_fit_unobserved=counts([r for r in decisions if r['native_fit'] is None]),
            direct_cap_denials=counts([r for r in denied if r['reason'] == 'cap']),
            cap_fifo_holds=counts([r for r in denied if r['reason'] == 'fifo_cap']),
            all_gate_denials=counts(denied),
            gate_denials_by_reason=dict(collections.Counter(r['reason'] for r in denied)),
            actual_waiting_allocation_failures=counts([r for r in allocations if not r['actual']]),
            semantics='Repeated evaluations are not distinct requests. FIFO rows have no checked fit. '
                'Native waiting allocation failures include recovery attempts; new-gate fit refusals '
                'do not. A gate denial is deferral, not dropped or failed service.'),
        actual_preemptions=dict(count=preemptions,
            evidence='recorded method-return events' if preemption_events is not None else
                     'raw count only' if preemptions is not None else 'unknown'),
        fixed_samples=[dict(source_index=i, request_id=requests[i]['request_id'],
            status=requests[i]['status'], finish_reason=requests[i].get('stop_reason'),
            native_stop_reason=requests[i]['native_stop_reason'],
            output_tokens=len(requests[i]['output_token_ids']), output_text=requests[i]['output_text'])
            for i in SAMPLE_INDICES],
        quality_scope='Structural counts and fixed source-order samples only; no task-quality scoring. '
            'Indices 0,1,64,128,256,383 were fixed before inspecting results. raw.requests retains '
            'measure_episode source insertion order. This is a new task baseline, not cross-task speedup.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite: '+str(args.output))
    result = summarize(args.cell.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with args.output.open('x') as stream:
        stream.write(payload)


if __name__ == '__main__':
    main()
