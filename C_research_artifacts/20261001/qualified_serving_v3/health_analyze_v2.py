#!/usr/bin/env python3
"""Read-only GSM8K health scoring for health_native.py; never loads a model.

The original sixteen requests remain the denominator. This is a development
screen and neither a benchmark score nor a test of cross-policy equivalence.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SOURCE_SHA = '6e51744ac762adab1fa3bf710efc413283f2a13cd3bb1a0560c841781d4dc11a'
NUMBERS = re.compile(r'[-+]?\d*\.\d+|\d+')
COMMA = re.compile(r'(\d),(\d)')
THRESHOLDS = dict(planned=16, completed_min=16, empty_max=0, natural_eos_min=12,
                  correct_min=8, correct_and_eos_min=8, periodic_suffix_max=1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction(text):
    values = NUMBERS.findall(COMMA.sub(r'\1\2', text))
    return values[-1] if values else None


def periodic(ids):
    if len(ids) < 256:
        return None
    tail = ids[-256:]
    return next((p for p in range(1, 17)
                 if all(tail[i] == tail[i-p] for i in range(p, 256))), None)


def token_ids(value, allow_empty=False):
    return isinstance(value, list) and (allow_empty or bool(value)) and all(
        type(v) is int and v >= 0 for v in value)


def eos_ids(receipt):
    values = []
    raw = [receipt.get('hf_eos_token_id')]
    generation = receipt.get('generation_config')
    if isinstance(generation, dict):
        raw.append(generation.get('eos_token_id'))
    for item in raw:
        if type(item) is int and item >= 0:
            values.append(item)
        elif isinstance(item, list) and all(type(v) is int and v >= 0 for v in item):
            values.extend(item)
    return sorted(set(values))



# Native vLLM may append eight lowercase hex characters while returning the
# original external ID in RequestOutput. Only that pinned transformation is
# accepted; no generic suffix stripping or many-to-one mapping is permitted.
IDENTITY_SOURCE = {
    "input_processor_sha256": "c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15",
    "input_processor_lines": [232, 240],
    "rule": 'request.request_id = f"{request.external_req_id}-{random_uuid():.8}"',
    "output_processor_sha256": "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b",
    "output_processor_lines": [370, 371],
    "scope": "Exact external ID or external ID plus '-' and exactly eight lowercase hex characters.",
}


def native_id_mapping(rows, expected_external_ids):
    """Return actual-to-external mapping and strict inventory errors."""
    expected = set(expected_external_ids)
    if not isinstance(rows, dict):
        return {}, ['native_sampling_inventory_not_object']
    mapping, seen, errors = {}, set(), []
    for actual in rows:
        external = None
        if isinstance(actual, str) and actual in expected:
            external = actual
        elif isinstance(actual, str):
            match = re.fullmatch(r"(.+)-([0-9a-f]{8})", actual)
            if match and match.group(1) in expected:
                external = match.group(1)
        if external is None:
            errors.append('native_sampling_unknown_id:' + str(actual))
            continue
        mapping[actual] = external
        if external in seen:
            errors.append('native_sampling_duplicate_external_id:' + external)
        seen.add(external)
    for missing in sorted(expected - seen):
        errors.append('native_sampling_missing_external_id:' + missing)
    return mapping, errors


def analyze(run, inputs):
    run, inputs = Path(run), Path(inputs)
    source_file = inputs / 'workload.json' if inputs.is_dir() else inputs
    if sha(source_file) != SOURCE_SHA:
        raise ValueError('The original frozen sixteen-question workload differs')
    source = json.loads(source_file.read_text())
    requests = source['requests']
    if len(requests) != 16 or [r['example_index'] for r in requests] != list(range(16)):
        raise ValueError('The frozen source-order request set differs')
    issues, hashes = [], {}

    def read(name, default):
        path = run / name
        if not path.is_file():
            issues.append('missing:' + name)
            return default
        hashes[name] = sha(path)
        try:
            return json.loads(path.read_text())
        except (ValueError, UnicodeError):
            issues.append('invalid_json:' + name)
            return default

    status = read('status.json', {})
    original = read('source-input.json', {})
    rendered = read('rendered-inputs.json', [])
    outputs = read('measured-outputs.json', [])
    config = read('config.json', {})
    tokenizer = read('tokenizer.json', {})
    eos = read('resolved-eos.json', {})
    native_sampling = read('measured-native-sampling.json', {})
    drain = read('native-drain.json', {})
    reset = read('prefix-cache-reset.json', {})
    provenance = read('provenance.json', {})
    runtime = read('runtime.json', {})
    if status.get('status') != 'COMPLETE' or status.get('expected_requests') != 16:
        issues.append('cell_incomplete')
    if original != source or provenance.get('input_sha256') != SOURCE_SHA:
        issues.append('source_or_provenance_mismatch')
    if drain.get('status') != 'QUALIFIED':
        issues.append('native_drain_unqualified')
    if not (reset.get('reset_succeeded') is True and reset.get('cached_hash_keys_after') == 0
            and reset.get('before', {}).get('status') == 'QUALIFIED'
            and reset.get('after', {}).get('status') == 'QUALIFIED'):
        issues.append('cold_apc_reset_unqualified')
    args, sampling = config.get('engine_args', {}), config.get('sampling', {})
    expected_sampling = dict(temperature=0.0, max_tokens=1024, min_tokens=0,
                             ignore_eos=False, stop=[], seed=20260905)
    if any(sampling.get(k) != v for k, v in expected_sampling.items()):
        issues.append('sampling_contract_mismatch')
    if not (args.get('enable_prefix_caching') is True
            and args.get('async_scheduling') is False
            and args.get('max_model_len') == 4096
            and args.get('max_num_seqs') == 32
            and args.get('max_num_batched_tokens') == 1024):
        issues.append('runtime_scope_mismatch')
    if not tokenizer.get('chat_template') or tokenizer.get('answer_cue') != 'Answer:':
        issues.append('tokenizer_template_receipt_missing')
    allowed_eos = eos_ids(eos)
    if not allowed_eos:
        issues.append('model_eos_unqualified')
    expected_native_ids = {'measured/' + r['request_id'] for r in requests}
    native_mapping, native_identity_issues = native_id_mapping(native_sampling, expected_native_ids)
    issues.extend(native_identity_issues)
    native_sampling = native_sampling if isinstance(native_sampling, dict) else {}
    for rid, observed in native_sampling.items():
        extra = observed.get('stop_token_ids') or []
        all_stop = observed.get('all_stop_token_ids') or []
        primary = observed.get('_eos_token_id')
        valid_ids = token_ids(extra, allow_empty=True) and token_ids(all_stop, allow_empty=True)
        if not (observed.get('stop') in (None, []) and observed.get('ignore_eos') is False
                and observed.get('min_tokens') == 0 and observed.get('max_tokens') == 1024
                and valid_ids and set(extra + all_stop).issubset(allowed_eos)
                and (primary in allowed_eos or bool(set(all_stop) & set(allowed_eos)))):
            issues.append(rid + ':native_eos_only_sampling_unqualified')
    cell_source = run / 'cell-source.py'
    if cell_source.is_file():
        hashes['cell-source.py'] = sha(cell_source)
        if provenance.get('cell_sha256') != hashes['cell-source.py']:
            issues.append('cell_source_provenance_mismatch')
    else:
        issues.append('missing:cell-source.py')

    def inventory(rows, name):
        if not isinstance(rows, list):
            issues.append(name + ':not_list')
            return {}
        index = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('request_id'), str):
                issues.append(name + ':invalid_row')
                continue
            rid = row['request_id']
            if rid in index:
                issues.append(name + ':duplicate:' + rid)
            index[rid] = row
        if set(index) != {r['request_id'] for r in requests}:
            issues.append(name + ':request_inventory_mismatch')
        return index

    render_by_id, output_by_id = inventory(rendered, 'rendered'), inventory(outputs, 'outputs')
    per_request = []
    identity_fields = ('request_id', 'example_index', 'question', 'gold', 'chat_user_content')
    for src in requests:
        rid, row_issues = src['request_id'], []
        render, row = render_by_id.get(rid), output_by_id.get(rid)
        if render is None:
            row_issues.append('missing_rendered_request')
        else:
            if any(render.get(k) != src[k] for k in identity_fields):
                row_issues.append('rendered_source_mismatch')
            if (not isinstance(render.get('prompt'), str)
                    or not render['prompt'].endswith('Answer:')
                    or not token_ids(render.get('prompt_token_ids'))
                    or len(render.get('prompt_token_ids', [])) + 1024 > 4096):
                row_issues.append('rendered_prompt_invalid')
        if row is None:
            per_request.append(dict(request_id=rid, example_index=src['example_index'],
                gold=src['gold'], status='MISSING', completed=False, correct=False,
                natural_eos=False, empty=True, output_tokens=0,
                periodic_suffix_period=None, prediction=None,
                issues=row_issues + ['missing_original_output']))
            continue
        if any(row.get(k) != src[k] for k in identity_fields):
            row_issues.append('output_source_mismatch')
        if render and any(row.get(k) != render.get(k) for k in ('prompt', 'prompt_token_ids')):
            row_issues.append('output_prompt_mismatch')
        ids, times, text = row.get('output_token_ids'), row.get('token_times_s'), row.get('output_text')
        if not token_ids(ids, allow_empty=True) or len(ids) > 1024:
            row_issues.append('invalid_output_ids')
            ids = []
        if not isinstance(text, str):
            row_issues.append('invalid_output_text')
            text = ''
        if (not isinstance(times, list) or len(times) != len(ids)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            row_issues.append('invalid_host_times')
        completed = row.get('finished') is True and row.get('finish_reason') in ('stop', 'length')
        if row.get('finished') is True and not completed:
            row_issues.append('unexpected_completion_reason')
        if row.get('finish_reason') == 'length' and len(ids) != 1024:
            row_issues.append('length_finish_without_cap')
        stop = row.get('stop_reason')
        natural = completed and row.get('finish_reason') == 'stop' and bool(allowed_eos) and (
            stop is None or (type(stop) is int and stop in allowed_eos))
        if row.get('finish_reason') == 'stop' and not natural:
            row_issues.append('unexplained_non_eos_stop')
        pred = prediction(text)
        correct = completed and not row_issues and pred == src['gold']
        per_request.append(dict(request_id=rid, example_index=src['example_index'],
            gold=src['gold'], status='COMPLETE' if completed else 'INCOMPLETE',
            completed=completed, correct=correct, natural_eos=natural,
            finish_reason=row.get('finish_reason'), stop_reason=stop,
            empty=not text.strip(), output_tokens=len(ids), output_text=text,
            prediction=pred, periodic_suffix_period=periodic(ids),
            host_completion_s=row.get('host_elapsed_s'), issues=row_issues))
        issues.extend(rid + ':' + value for value in row_issues)

    counts = dict(planned=16, completed=sum(r['completed'] for r in per_request),
        empty=sum(r['empty'] for r in per_request),
        natural_eos=sum(r['natural_eos'] for r in per_request),
        correct=sum(r['correct'] for r in per_request),
        correct_and_eos=sum(r['correct'] and r['natural_eos'] for r in per_request),
        periodic_suffix=sum(r['periodic_suffix_period'] is not None for r in per_request),
        output_tokens=sum(r['output_tokens'] for r in per_request),
        length_capped=sum(r.get('finish_reason') == 'length' for r in per_request))
    if counts['completed'] != 16:
        issues.append('not_all_planned_requests_completed')
    gates = dict(completed=counts['completed'] == 16, empty=counts['empty'] == 0,
        natural_eos=counts['natural_eos'] >= 12, correct=counts['correct'] >= 8,
        correct_and_eos=counts['correct_and_eos'] >= 8,
        periodic_suffix=counts['periodic_suffix'] <= 1)
    valid = not issues
    return dict(schema='qualified-serving-v3-gsm-health-analysis-v2',
        analysis_correction='Only native/external request-ID alignment changed from v1; outputs, validity gates, denominator, numeric scoring and quality thresholds are unchanged.',
        identity_source=IDENTITY_SOURCE, native_to_external_request_ids=native_mapping,
        validity='COMPLETE' if valid else 'INVALID_OR_INCOMPLETE',
        screening=('PASS_DEVELOPMENT_SCREEN' if all(gates.values()) else 'FAIL_DEVELOPMENT_SCREEN')
                  if valid else 'NOT_INTERPRETABLE',
        issues=issues, thresholds=THRESHOLDS, gates=gates, counts=counts,
        allowed_model_eos_ids=allowed_eos, model_directory=provenance.get('model_dir'),
        runtime=runtime, engine_args=args, sampling=sampling, per_request=per_request,
        finish_counts=dict(Counter(r.get('finish_reason') or 'missing_or_unfinished' for r in per_request)),
        original_file_sha256=hashes, source_workload_sha256=SOURCE_SHA,
        analyzer_sha256=sha(__file__),
        limits=['All sixteen original source-order questions remain the denominator.',
                'Thresholds screen further development, not the original OLMoE completion verdict.',
                'No semantic quality equivalence, full GSM8K benchmark, pressure or performance claim.',
                'Host delivery times do not expose within-chunk inter-token latency.',
                'Model weight integrity and shared-lock lifecycle require the parent launcher receipts.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path,
        default=Path(__file__).resolve().parent.parent / '20261001_c_instruct_gsm8k_inputs_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.inputs)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('validity', 'screening', 'counts')}, ensure_ascii=False))
    raise SystemExit(2 if result['validity'] != 'COMPLETE' else 0)
