#!/usr/bin/env python3
"""Score A raw.json captures against independent GSM8K input references.

Uses only local files and a hash-pinned tokenizer. Every planned request stays
in the denominator. No output changes, last-number fallback in the primary
score, or cap/EOS conflation. Legacy extraction is reported separately.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
from collections import Counter
from fractions import Fraction

ROOT = Path(__file__).resolve().parent
NUMBER = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?|[-+]?\.\d+'
MARKED = re.compile(r'(?:####|(?:final\s+)?answer\s*(?:is|:|=))\s*\$?\s*(' + NUMBER + r')(?![\w/]|\.\d)', re.I)
BOXED = re.compile(r'\\boxed\{\s*(' + NUMBER + r')\s*\}')
LEGACY_NUMBERS = re.compile(r'[-+]?\d*\.\d+|\d+')


def normalize_number(value):
    value = re.sub(r'\s+', '', value).replace(',', '')
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[-+]?\d+(?:\.\d+)?)?', value):
        raise ValueError('not a supported exact numeric answer')
    parts = value.split('/')
    answer = Fraction(parts[0])
    if len(parts) == 2:
        answer /= Fraction(parts[1])
    return str(answer.numerator) if answer.denominator == 1 else str(answer)


def strict_answer(text):
    candidates = [(m.start(), m.group(1)) for pattern in (MARKED, BOXED) for m in pattern.finditer(text)]
    if not candidates:
        return None
    try:
        return normalize_number(max(candidates)[1])
    except (ValueError, ZeroDivisionError):
        return None


def legacy_answer(text):
    numbers = LEGACY_NUMBERS.findall(re.sub(r'(\d),(\d)', r'\1\2', text))
    return numbers[-1] if numbers else None


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None, maximum=None)
    def q(fraction):
        position = (len(values) - 1) * fraction
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return dict(n=len(values), mean=statistics.mean(values), p50=q(.5), p95=q(.95), maximum=values[-1])


def evaluate(directory, workload, prepared_config, references, tokenizer):
    raw, config = read(directory / 'raw.json'), read(directory / 'config.json')
    expected = {r['request_id']: r for r in workload['source_requests']}
    if config['model'] != prepared_config['model'] or config.get('workload_sha256') != prepared_config['workload_sha256']:
        raise ValueError('Run model/workload identity differs from prepared input')
    if config.get('ignore_eos') is not False or config.get('min_tokens') != 0:
        raise ValueError('Run does not use the natural-EOS sampling contract')
    cap = prepared_config['output_tokens']
    if config.get('output_tokens') != cap or config.get('output_tokens_by_request', {}) != {}:
        raise ValueError('Run output cap differs from frozen inputs')
    actual = {r['request_id']: r for r in raw['requests']}
    if len(actual) != len(raw['requests']) or set(actual) - set(expected):
        raise ValueError('Duplicate or unknown raw request ID')
    event_map = {rid: [] for rid in expected}
    for event in raw.get('output_events', []):
        if event['request_id'] not in expected:
            raise ValueError('Unknown output-event request ID')
        event_map[event['request_id']].append(event)
    rows = []
    for source, arrival in zip(workload['source_requests'], workload['arrival_traces_s']['steady']):
        rid = source['request_id']
        row = actual.get(rid, dict(status='missing', output_token_ids=[], token_times_s=[]))
        ids, times = row['output_token_ids'], row['token_times_s']
        if (len(ids) != len(times) or len(ids) > cap
                or any(type(t) is not int or t < 0 for t in ids)
                or any(not math.isfinite(t) or t < arrival for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError(f'{rid}: invalid token/timing capture')
        if rid in actual and (row.get('prompt_token_ids_sha256') != source['prompt_token_ids_sha256']
                              or row.get('arrival_s') != arrival or row.get('max_output_tokens') != cap):
            raise ValueError(f'{rid}: prompt/arrival/cap identity mismatch')
        finished = row['status'] == 'completed'
        terminal = [e for e in event_map[rid] if e.get('finished')]
        event_reason = terminal[-1].get('finish_reason') if terminal else None
        # Current A capture stores completion.finish_reason in row.stop_reason.
        # It is not the native EOS token/string identifier.
        reason = row.get('finish_reason', event_reason)
        legacy_row_reason = row.get('stop_reason')
        if reason is None and legacy_row_reason in ('stop', 'length'):
            reason = legacy_row_reason
        if finished and (reason not in ('stop', 'length') or (reason == 'length' and len(ids) != cap)):
            raise ValueError(f'{rid}: inconsistent completion/cap')
        completion = row.get('completion_s')
        if finished and (completion is None or completion < arrival or (times and completion < times[-1])):
            raise ValueError(f'{rid}: invalid completion boundary')
        positives = [e for e in event_map[rid] if e.get('chunk_size', 0) > 0]
        if sum(e['chunk_size'] for e in positives) != len(ids):
            raise ValueError(f'{rid}: output events do not account for captured tokens')
        receipt_times = [e['received_s'] for e in positives]
        gaps = [b - a for a, b in zip(receipt_times, receipt_times[1:])]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        answer, legacy = strict_answer(text), legacy_answer(text)
        gold = references[rid]['gold']
        correct = finished and answer == normalize_number(gold)
        rows.append(dict(request_id=rid, gold=gold, output_text=text, answer=answer,
                         strict_correct=bool(correct), legacy_answer=legacy,
                         legacy_correct=finished and legacy == gold,
                         correct_and_natural_stop=bool(correct and reason == 'stop'),
                         completed=finished, raw_status=row['status'], finish_reason=reason,
                         stop_identifier=row.get('native_stop_reason'),
                         stop_identifier_available='native_stop_reason' in row,
                         natural_stop_inferred=finished and reason == 'stop',
                         cap_truncated=finished and reason == 'length',
                         reached_cap=len(ids) == cap, output_tokens=len(ids), output_token_ids=ids,
                         flow_s=completion-arrival if finished else None,
                         ttft_s=times[0]-arrival if times else None,
                         max_host_output_gap_s=max(gaps) if gaps else None,
                         token_level_resolved=all(e['chunk_size'] == 1 for e in positives)))
    n = len(expected)
    wall = raw['observation_end_s']
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError('Invalid full-episode timing denominator')
    token_count = sum(r['output_tokens'] for r in rows)
    all_complete = raw['status'] == 'COMPLETE' and all(r['completed'] for r in rows)
    return dict(status=raw['status'], all_complete=all_complete, requests=n,
                missing_requests=sorted(set(expected)-set(actual)),
                completed=sum(r['completed'] for r in rows),
                strict_accuracy=sum(r['strict_correct'] for r in rows)/n,
                legacy_accuracy=sum(r['legacy_correct'] for r in rows)/n,
                correct_and_natural_stop=sum(r['correct_and_natural_stop'] for r in rows),
                natural_stop_count=sum(r['natural_stop_inferred'] for r in rows),
                cap_truncated_count=sum(r['cap_truncated'] for r in rows),
                reached_cap_count=sum(r['reached_cap'] for r in rows),
                finish_counts=dict(Counter(r['finish_reason'] or r['raw_status'] for r in rows)),
                output_tokens=token_count, episode_wall_s=wall, actual_output_tokens_per_s=token_count/wall,
                flow=distribution([r['flow_s'] for r in rows if r['flow_s'] is not None]),
                ttft=distribution([r['ttft_s'] for r in rows if r['ttft_s'] is not None]),
                request_max_host_output_gap=distribution([r['max_host_output_gap_s'] for r in rows if r['max_host_output_gap_s'] is not None]),
                raw_sha256=digest(directory/'raw.json'), per_request=rows)


def compare(a, b):
    pairs = list(zip(a['per_request'], b['per_request']))
    if any(x['request_id'] != y['request_id'] for x, y in pairs):
        raise ValueError('Pair request order differs')
    return dict(identical_token_sequences=sum(x['output_token_ids'] == y['output_token_ids'] for x, y in pairs),
                changed_answers=sum(x['answer'] != y['answer'] for x, y in pairs),
                left_only_correct=[x['request_id'] for x,y in pairs if x['strict_correct'] and not y['strict_correct']],
                right_only_correct=[x['request_id'] for x,y in pairs if y['strict_correct'] and not x['strict_correct']],
                changed_finish_reason=[x['request_id'] for x,y in pairs if x['finish_reason'] != y['finish_reason']],
                accuracy_delta=a['strict_accuracy']-b['strict_accuracy'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='Prepared inputs directory containing config/workload')
    p.add_argument('--tokenizer-dir', type=Path, required=True)
    p.add_argument('--run', action='append', required=True, help='NAME=/path/to/A/output; repeat for matched arms')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Refusing to replace an existing quality report')
    config, work = read(args.inputs/'config.json'), read(args.inputs/'workload.json')
    if hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest() != config['workload_sha256']:
        raise ValueError('Prepared workload identity changed')
    if (digest(ROOT/'reference_answers.json') != config['reference_answers_sha256']
            or digest(ROOT/'tasks_text.json') != config['tasks_text_sha256']):
        raise ValueError('Input questions/reference answers changed')
    hashes = {name: digest(args.tokenizer_dir/name) for name in config['tokenizer_files_sha256']}
    if hashes != config['tokenizer_files_sha256']:
        raise ValueError('Output tokenizer differs from prepared model tokenizer')
    if work['sampling'] != dict(ignore_eos=False, min_tokens=0, max_tokens=config['output_tokens'],
                                temperature=0.0, stop=[], stop_token_ids=[]):
        raise ValueError('Prepared generation contract is not natural EOS without extra stops')
    from tokenizer_backend import OfflineTokenizer
    tokenizer = OfflineTokenizer(args.tokenizer_dir)
    refs = {r['request_id']: r for r in read(ROOT/'reference_answers.json')['references']}
    cells = {}
    for specification in args.run:
        name, separator, path = specification.partition('=')
        if not separator or not name or name in cells:
            raise ValueError('Each run needs a unique NAME=PATH')
        cells[name] = evaluate(Path(path), work, config, refs, tokenizer)
    result = dict(schema='a-healthy-quality-v1', cells=cells,
                  pairs=[dict(left=a, right=b, **compare(cells[a],cells[b])) for a,b in itertools.combinations(cells,2)],
                  notes=['Primary score: exact normalized numeric answer after an explicit answer marker or boxed answer; legacy last-number string match is separate.',
                         'All planned requests remain in the denominator; incomplete/missing requests score zero.',
                         'A length finish can score correctly, but is cap-truncated and excluded from correct_and_natural_stop.',
                         'stop is inferred EOS-compatible termination only under no-extra-stop sampling; the old A capture does not retain native stop identifiers.',
                         'Output content/length can change by scheduler; rates use actual tokens/full episode wall and are not equal-work speedups.',
                         'Gap is between positive-output host receipts; TTFT excluded; no interpolation within multi-token chunks.',
                         '16/32 tasks are exploratory quality checks, not quality equivalence or statistical robustness evidence.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
