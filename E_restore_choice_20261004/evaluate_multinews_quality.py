"""CPU-only frozen MultiNews scoring. No group means metric checks only, GPU_UNRUN.

Dependencies are read from an explicitly supplied isolated directory, never installed.
Completed local episodes must match the entire frozen input and runtime workload hash.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from evaluate_short_quality import load_tokenizers

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / 'inputs_multinews_candidate'
LOCKED = {
    'pro6000_multinews_candidate_256.json': '00fa8875af4fa3fde61d31dff503fda9216f98839ed57d9db58ac2480a1c30c6',
    'quality_contract.json': '65b6b3a4c52cb2bc0f6b29b5b43858db001f6ec47bc0de0a265166c25da7e101',
    'sources/multi_news_selected.json': 'c6f49f4d97ffc9aa5f1ff2c5d50bffdf31ba7fb12cf7256b4124f1574326782c',
    'sources/official_metrics.py': 'e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323',
    'sources/official_eval.py': '1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721',
}
PROFILES = {
    'multinews256': dict(path=CANDIDATE/'pro6000_multinews_candidate_256.json',
        sha256=LOCKED['pro6000_multinews_candidate_256.json'], requests=256, long_requests=192,
        max_num_seqs=256),
    'multinews320': dict(path=ROOT/'inputs_multinews320/pro6000_multinews_candidate_320.json',
        sha256='21c601c5aa8bda312ba3a4f10c88b8cc34bbecfbc2fb15f98daf2ac53f707995',
        requests=320, long_requests=256, max_num_seqs=320,
        addendum_path=ROOT/'inputs_multinews320/quality_scale_addendum.json',
        addendum_sha256='7add92c6d87ec7e38fca4b73013e11d6dc6743791d3e49e4c54a6db311339bc3'),
}


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_workload_profile(name):
    """Bind one explicit frozen input; the original metric contract is never edited."""
    profile = PROFILES[name]
    require(digest(profile['path']) == profile['sha256'], 'Selected workload hash mismatch')
    workload = read(profile['path'])
    rows = workload['source_requests']
    require(len(rows) == profile['requests'], 'Frozen workload count mismatch')
    require(Counter(r['request_class'] for r in rows) ==
            {'short': 64, 'long': profile['long_requests']}, 'Frozen task counts mismatch')
    longs = [r for r in rows if r['request_class'] == 'long']
    order = workload['source_manifest']['selected_source_indices']
    require(len(order) == len(set(order)) == 27, 'Expected the same 27 source bundles')
    require([r['example_index'] for r in longs] == [order[i % 27] for i in range(len(longs))],
            'Long source cycle changed')
    require(Counter(str(r['example_index']) for r in longs) ==
            workload['statistics']['long_repetitions_by_source'], 'Frozen source multiplicity mismatch')
    require(workload['suggested_fixed_resources']['max_num_seqs'] == profile['max_num_seqs'],
            'Frozen max_num_seqs mismatch')
    addendum = None
    if name == 'multinews320':
        require(digest(profile['addendum_path']) == profile['addendum_sha256'], 'Scale addendum hash mismatch')
        addendum = read(profile['addendum_path'])
        parent_profile = PROFILES['multinews256']
        require(digest(parent_profile['path']) == parent_profile['sha256'], 'Parent workload hash mismatch')
        parent = read(parent_profile['path'])
        require(rows[:256] == parent['source_requests'], 'Original 256-request prefix changed')
        require(len(rows[256:]) == 64 and all(r['request_class'] == 'long' for r in rows[256:]),
                'Expected exactly 64 appended long requests')
        require(order == parent['source_manifest']['selected_source_indices'] and rows[256]['example_index'] == 32,
                'Appended long cycle must continue at ordinal192/source32')
        require(addendum['extends_contract_sha256'] == LOCKED['quality_contract.json'] and
                addendum['workload_sha256'] == profile['sha256'] and
                addendum['parent_workload_sha256'] == parent_profile['sha256'], 'Scale provenance mismatch')
        require(addendum['requests'] == len(rows) and addendum['long_requests'] == len(longs) and
                addendum['short_requests'] == 64 and addendum['source_bundles'] == 27 and
                addendum['long_source_cycle'] == order and
                addendum['long_repetitions_by_source'] == workload['statistics']['long_repetitions_by_source'] and
                addendum['fixed_resources'] == workload['suggested_fixed_resources'], 'Scale scope mismatch')
    return workload, dict(name=name, path=str(profile['path']), sha256=profile['sha256'],
        requests=len(rows), long_requests=len(longs), max_num_seqs=profile['max_num_seqs'],
        original_256_prefix_checked=name == 'multinews320', source_cycle_checked=True,
        addendum_sha256=profile.get('addendum_sha256'), scale_addendum=addendum)


def official_function(path, name, namespace):
    """Compile the unmodified pinned function AST; omit unrelated Chinese metrics imports."""
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def load_rouge(site, receipt_path):
    receipt = read(receipt_path)
    require([(p['name'], p['version']) for p in receipt['packages']] ==
            [('rouge', '1.0.1'), ('six', '1.17.0')], 'Unexpected dependency versions')
    for package in receipt['packages']:
        for filename, expected in package['files'].items():
            require(digest(site / filename) == expected, f'Dependency changed: {filename}')
    # Explicit dependency path precedes environment packages, with origin checks afterwards.
    sys.path.insert(0, str(site.resolve()))
    import rouge
    import six
    for module, version in [(rouge, '1.0.1'), (six, '1.17.0')]:
        require(module.__version__ == version, f'Unexpected {module.__name__} version')
        require(Path(module.__file__).resolve().is_relative_to(site.resolve()),
                f'{module.__name__} was imported outside the explicit isolated dependency path')
    return rouge.Rouge, dict(receipt_sha256=digest(receipt_path), receipt=receipt,
        loaded_rouge_path=str(rouge.__file__), loaded_six_path=str(six.__file__))


def score_reference(prediction, reference, Rouge):
    """Same default get_scores call as LongBench, with exceptions exposed as invalid."""
    try:
        value = Rouge().get_scores([prediction], [reference], avg=True)['rouge-l']['f']
        require(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1,
                'Invalid numerical ROUGE-L F1')
        return dict(valid=True, score_f=float(value), official_compatible_f=float(value), error=None)
    except Exception as exc:
        # Preserve official numerical behavior only as a labeled diagnostic, never as a valid zero.
        return dict(valid=False, score_f=None, official_compatible_f=0.0,
                    error=dict(type=type(exc).__name__, message=str(exc)))


def score_output(prediction, references, Rouge):
    require(bool(references) and all(isinstance(r, str) and r.strip() for r in references),
            'Reference summaries must be nonempty strings')
    scores = [score_reference(prediction, reference, Rouge) for reference in references]
    valid = all(s['valid'] for s in scores)
    return dict(valid=valid, reference_scores=scores,
        rouge_l_f_percent=100 * max(s['score_f'] for s in scores) if valid else None,
        official_compatible_percent=100 * max(s['official_compatible_f'] for s in scores))


def aggregate(rows):
    bundles = defaultdict(list)
    for row in rows:
        bundles[row['source_index']].append(row)
    valid = all(row['score']['valid'] and not row.get('generation_error') for row in rows)
    bundle_scores = {str(key): dict(requests=len(values),
        valid=all(v['score']['valid'] and not v.get('generation_error') for v in values),
        official_compatible_percent=statistics.mean(v['score']['official_compatible_percent'] for v in values))
        for key, values in sorted(bundles.items())}
    macro = statistics.mean(v['official_compatible_percent'] for v in bundle_scores.values())
    weighted = statistics.mean(row['score']['official_compatible_percent'] for row in rows)
    return dict(valid=valid, requests=len(rows), bundles=len(bundles),
        primary_bundle_macro_rouge_l_percent=macro if valid else None,
        secondary_request_weighted_rouge_l_percent=weighted if valid else None,
        official_compatible_diagnostic=dict(bundle_macro_percent=macro, request_weighted_percent=weighted,
            reportable_as_valid_quality=valid,
            note='Includes all rows. Exceptions retain official zero only as an explicitly invalid diagnostic.'),
        per_bundle=bundle_scores,
        metric_error_requests=sum(not r['score']['valid'] for r in rows),
        generation_error_requests=sum(bool(r.get('generation_error')) for r in rows))


def cpu_check(Rouge):
    official = official_function(CANDIDATE / 'sources/official_metrics.py', 'rouge_score', {'Rouge': Rouge})
    scorer = official_function(CANDIDATE / 'sources/official_eval.py', 'scorer',
                               {'dataset2metric': {'multi_news': official}})
    cases = [('exact', 'The council approved the bridge.', 'The council approved the bridge.'),
             ('partial', 'The council approved the bridge.', 'The mayor opened the bridge.'),
             ('no_overlap', 'alpha beta gamma', 'delta epsilon zeta'),
             ('case_punctuation', 'Bridge OPENED, Monday.', 'bridge opened Monday.'),
             ('multiple_sentences', 'A bridge opened. Trains resumed.', 'Trains resumed. A bridge opened.'),
             ('repetition', 'bridge bridge bridge opened opened', 'bridge opened'),
             ('whitespace', ' A  bridge\nopened. ', 'A bridge opened.'),
             ('empty', '', 'The bridge opened.'),
             ('punctuation_only', '...', 'The bridge opened.')]
    results = []
    for name, prediction, reference in cases:
        direct = score_reference(prediction, reference, Rouge)
        expected = official(prediction, reference)
        require(abs(expected - direct['official_compatible_f']) <= 1e-12, f'Official mismatch: {name}')
        if name in ('empty', 'punctuation_only'):
            require(not direct['valid'] and direct['score_f'] is None, f'Failure hidden: {name}')
        else:
            require(direct['valid'], f'Unexpected metric error: {name}')
        results.append(dict(name=name, prediction=prediction, reference=reference,
                            official_f=expected, cli=direct, agreement=True))
    prediction, references = 'The bridge opened.', ['No trains returned.', 'The bridge opened.']
    multiple = score_output(prediction, references, Rouge)
    expected = scorer('multi_news', [prediction], [references], [])
    require(round(multiple['rouge_l_f_percent'], 2) == expected, 'Multi-reference max mismatch')
    class BrokenRouge:
        def get_scores(self, *unused, **kwargs):
            raise RuntimeError('CPU_CHECK_INJECTED_FAILURE')
    broken = score_reference('some text', 'some reference', BrokenRouge)
    require(not broken['valid'] and broken['score_f'] is None and broken['error']['type'] == 'RuntimeError',
            'Injected failure was not made invalid')
    small = [dict(source_index=0, score=dict(valid=True, official_compatible_percent=50.0))] * 8
    small += [dict(source_index=1, score=dict(valid=True, official_compatible_percent=100.0))] * 7
    aggregation = aggregate(small)
    require(aggregation['primary_bundle_macro_rouge_l_percent'] == 75, 'Wrong bundle macro')
    require(abs(aggregation['secondary_request_weighted_rouge_l_percent'] - 1100/15) < 1e-12,
            'Wrong request-weighted mean')
    bad_aggregation = aggregate([dict(source_index=0, score=dict(valid=False, official_compatible_percent=0))])
    require(bad_aggregation['primary_bundle_macro_rouge_l_percent'] is None, 'Invalid aggregate became a score')
    return dict(status='CPU_FORMULA_CHECK_PASS_NOT_GPU_QUALITY', fixed_cases=results,
        multi_reference=dict(cli=multiple, official_scorer_percent=expected, agreement=True),
        injected_failure=broken, aggregation_fixture=aggregation,
        invalid_aggregation_suppressed=True,
        source_loading='Unmodified AST function nodes from pinned official metrics/eval; unrelated imports omitted.')


def validate_episode(cell, group, frozen, profile):
    require(read(group / 'status.json').get('status') == 'COMPLETE', 'Group is not COMPLETE')
    require(read(cell / 'status.json').get('status') == 'COMPLETE', f'Cell is not COMPLETE: {cell}')
    inputs, config = read(cell / 'inputs.json'), read(cell / 'config.json')
    require(inputs == frozen, f'Episode inputs do not equal the complete frozen MultiNews workload: {cell}')
    require(config.get('natural') is True and config.get('requests') == profile['requests'],
            f"Episode must use all {profile['requests']} frozen requests and natural EOS")
    require(config.get('max_num_seqs') == profile['max_num_seqs'], 'Episode max_num_seqs mismatch')
    if profile['requests'] == 320:
        require(config.get('batch_tokens') == 2048 and config.get('host_gib') == 32 and
                config.get('kv_bytes') is None, '320 episode resource configuration mismatch')
        engine = read(cell / 'engine_args.json')
        required = dict(dtype='bfloat16', max_model_len=4096, max_num_seqs=320,
                        max_num_batched_tokens=2048, gpu_memory_utilization=0.9,
                        kv_cache_memory_bytes=None, kv_offloading_size=32)
        require(all(engine.get(key) == value for key, value in required.items()),
                '320 engine resources differ from the frozen scale addendum')
    hashes_path = next((parent / 'runtime_source_hashes.json' for parent in (cell, cell.parent)
                        if (parent / 'runtime_source_hashes.json').is_file()), None)
    require(hashes_path is not None, 'Missing runtime source hashes')
    require(read(hashes_path).get(config.get('workload')) == profile['sha256'],
            'Runtime workload hash does not match the frozen workload')
    raw = read(cell / 'raw.json')
    requests = raw['requests']
    require(len(requests) == len(frozen) == profile['requests'], 'Raw request count mismatch')
    require(len({r['request_id'] for r in requests}) == profile['requests'], 'Duplicate internal request identities')
    for index, (request, source) in enumerate(zip(requests, frozen)):
        expected = dict(external_id=f'measured/E{index:03d}', source_index=source['example_index'],
            gold=source['gold'], prompt_tokens=len(source['prompt_token_ids']),
            max_output_tokens=source['output_tokens'], arrival_s=source['arrival_s'], completed=True)
        for field, value in expected.items():
            require(request.get(field) == value, f'Raw/input mismatch at {index}: {field}')
        tokens = request['output_token_ids']
        require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens), 'Invalid token IDs')
        require(len(tokens) <= source['output_tokens'], 'Output exceeds cap')
        require(len(tokens) == len(request['token_times_s']), 'Output/token-time count mismatch')
        require(request.get('finish_reason') is not None and request.get('completion_s') is not None,
                'Missing completed request fields')
    return requests, config, dict(raw_sha256=digest(cell / 'raw.json'),
        inputs_sha256=digest(cell / 'inputs.json'), config_sha256=digest(cell / 'config.json'),
        status_sha256=digest(cell / 'status.json'), runtime_source_hashes_sha256=digest(hashes_path))


def evaluate(args, Rouge, workload, profile):
    group = args.group.resolve()
    paths = sorted(group.rglob('raw.json')) if group.is_dir() else []
    require(paths, 'No local completed raw episodes; GPU_UNRUN cannot be scored')
    require((group / 'status.json').is_file(), 'Missing group status')
    sources = {r['source_index']: r['source'] for r in read(CANDIDATE / 'sources/multi_news_selected.json')}
    binding = load_tokenizers(args.tokenizers_extension)
    require(binding.__version__ == '0.23.2', 'Tokenizers version must match frozen CPU validation')
    require(digest(args.tokenizer) == workload['source_manifest']['verified_inputs']['inputs_pro6000/sources/tokenizer.json'],
            'Tokenizer hash mismatch')
    tokenizer = binding.Tokenizer.from_file(str(args.tokenizer))
    frozen = workload['source_requests']
    # Validate official gold and complete source context before looking at predictions.
    template = workload['summary_instruction_template']
    for row in frozen:
        if row['task'] != 'multi_news':
            continue
        source = sources[row['example_index']]
        require(row['document_id'] == source['_id'] and
                row['source_context_sha256'] == sha(source['context'].encode()), 'Source identity mismatch')
        require(row['gold'] == row['reference_answer'] == source['answers'], 'Official gold mismatch')
        require(row['prompt'] == '<|endoftext|><|user|>\n' + template.format(**source) + '\n<|assistant|>\n',
                'Full official context/template mismatch')
    arms, audit = {}, {}
    for path in paths:
        cell = path.parent
        name = str(cell.relative_to(group))
        requests, config, provenance = validate_episode(cell, group, frozen, profile)
        rows = []
        for request, inp in zip(requests, frozen):
            if inp['task'] != 'multi_news':
                continue
            require(all(t < tokenizer.get_vocab_size(with_added_tokens=True) for t in request['output_token_ids']),
                    'Output token outside frozen vocabulary')
            text = tokenizer.decode(request['output_token_ids'], skip_special_tokens=True)
            references = sources[inp['example_index']]['answers']
            rows.append(dict(external_id=request['external_id'], input_request_id=inp['request_id'],
                source_index=inp['example_index'], source_id=inp['document_id'],
                repetition_index=inp['repetition_index'], text=text, reference_answers=references,
                output_token_ids=request['output_token_ids'], output_tokens=len(request['output_token_ids']),
                output_words=len(text.split()), finish_reason=request['finish_reason'], stop_reason=request.get('stop_reason'),
                empty_output=not text.strip(), length_stopped=request['finish_reason'] == 'length',
                generation_error=request['finish_reason'] not in ('stop', 'length'),
                score=score_output(text, references, Rouge)))
        multiplicity = Counter(str(r['source_index']) for r in rows)
        require(multiplicity == workload['statistics']['long_repetitions_by_source'] and
                len(rows) == profile['long_requests'],
                'Missing/repeated source requests')
        summary = aggregate(rows)
        summary.update(finish_reasons=dict(Counter(r['finish_reason'] for r in rows)),
            empty_outputs=sum(r['empty_output'] for r in rows),
            output_tokens=dict(min=min(r['output_tokens'] for r in rows), max=max(r['output_tokens'] for r in rows),
                               mean=statistics.mean(r['output_tokens'] for r in rows)),
            output_words=dict(min=min(r['output_words'] for r in rows), max=max(r['output_words'] for r in rows),
                              mean=statistics.mean(r['output_words'] for r in rows)))
        arms[name] = dict(policy=config.get('policy'), summary=summary, rows=rows, provenance=provenance)
    names = list(arms)
    paired = []
    for first_index, first in enumerate(names):
        for second in names[first_index+1:]:
            a, b = arms[first]['summary'], arms[second]['summary']
            valid = a['valid'] and b['valid']
            paired.append(dict(first=first, second=second, difference_direction='second-minus-first', valid=valid,
                primary_macro_difference=b['primary_bundle_macro_rouge_l_percent']-a['primary_bundle_macro_rouge_l_percent']
                    if valid else None,
                per_bundle_difference={key: b['per_bundle'][key]['official_compatible_percent']-
                    a['per_bundle'][key]['official_compatible_percent'] for key in a['per_bundle']} if valid else None))
    return dict(status='COMPLETE_VALID_QUALITY' if all(a['summary']['valid'] for a in arms.values())
                else 'INVALID_METRIC_OR_GENERATION_ERRORS', group=str(group), group_status_sha256=digest(group/'status.json'),
        tokenizer=dict(version=binding.__version__, path=str(binding.__file__), file_sha256=digest(Path(binding.__file__)),
                       tokenizer_sha256=digest(args.tokenizer), skip_special_tokens=True),
        arms=arms, paired=paired,
        identity_limit='Binding uses saved inputs and trusted runner runtime hashes; raw has no independent prompt-token hash.',
        manual_audit='Not automated: frozen source indices 0/120/194, first Host arm selection, at most six identities.',
        short_task='Not scored here; use the existing frozen GSM8K CLI separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path, nargs='?')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--dependency-receipt', type=Path, default=ROOT/'quality_tools/rouge_dependency_receipt.json')
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'inputs_pro6000/sources/tokenizer.json')
    parser.add_argument('--tokenizers-extension', type=Path)
    parser.add_argument('--workload-profile', choices=tuple(PROFILES), default='multinews256')
    args = parser.parse_args()
    result = dict(schema='E.multinews_quality.v1', status='INVALID_SETUP', arms={}, paired=[])
    exit_code = 1
    try:
        for filename, expected in LOCKED.items():
            require(digest(CANDIDATE / filename) == expected, f'Frozen source changed: {filename}')
        workload, workload_binding = load_workload_profile(args.workload_profile)
        Rouge, dependencies = load_rouge(args.deps, args.dependency_receipt)
        result.update(provenance=dict(locked_files_sha256=LOCKED, scorer_sha256=digest(Path(__file__)),
            dependencies=dependencies, workload_profile=workload_binding),
            metric_definition=read(CANDIDATE/'quality_contract.json'),
            metric_scale_addendum=workload_binding['scale_addendum'],
            cpu_check=cpu_check(Rouge))
        if args.group is None:
            result.update(status='CPU_VERIFIED_GPU_UNRUN', reason='No completed matching GPU group supplied; no quality scores.')
            exit_code = 0
        else:
            result.update(evaluate(args, Rouge, workload, PROFILES[args.workload_profile]))
            exit_code = int(result['status'] != 'COMPLETE_VALID_QUALITY')
    except Exception as exc:
        result.update(status='INVALID_INPUT_OR_SETUP', error=dict(type=type(exc).__name__, message=str(exc)))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], output=str(args.out), error=result.get('error'),
        summary={name: arm['summary'] for name, arm in result['arms'].items()}), ensure_ascii=False, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
