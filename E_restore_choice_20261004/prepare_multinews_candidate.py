"""Freeze one CPU-only MultiNews candidate; does not run or change an experiment."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re


EXPECTED = {
    'inputs_pro6000/pro6000_high_256.json': '6b2a65dc10ca32cc1d8b84e9c7c83e90796cb5e8eed1f1f0e0d98301b431c88b',
    'inputs_pro6000/sources/tokenizer.json': 'b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167',
    'inputs_multinews_candidate/sources/multi_news.jsonl': 'f8817c00af317c2d7f81128f334bcfe5780c6c2a515d5c96811aa0ddf8b5ee6b',
    'inputs_multinews_candidate/sources/official_dataset2prompt.json': '56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30',
    'inputs_multinews_candidate/sources/official_metrics.py': 'e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323',
    'inputs_multinews_candidate/sources/official_eval.py': '1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721',
    'inputs_multinews_candidate/sources/source_receipt.json': '09725756f57887733dfac1003d7232ff83aec8fff8b1ae9926fe7cdc528a00c4',
    'inputs_multinews_candidate/sources/official_eval_source_receipt.json': '76edfd3228caba8024609a799a02df6d8f29175c1f07c32b08cb3888a2c6b0e2',
}
ELIGIBLE = [0, 6, 16, 32, 34, 54, 57, 63, 93, 97, 99, 101, 112, 120, 122, 123,
            127, 137, 150, 153, 159, 165, 171, 184, 188, 191, 194]
QUALITY = {
    'status': 'FROZEN_BEFORE_GENERATION_GPU_UNRUN',
    'reference': 'Each selected official record answers list, stored verbatim; never supplied to model or selector.',
    'per_output_score': 'Official LongBench multi_news ROUGE-L F1: max over provided references, multiplied by 100.',
    'implementation': 'Pinned sources/official_metrics.py rouge_score and sources/official_eval.py scorer; '
        'python rouge.Rouge.get_scores([prediction], [reference], avg=True)[rouge-l][f]; scorer exceptions yield 0.',
    'prediction': 'Decode the entire generated continuation with the frozen tokenizer, skip_special_tokens=True; '
        'do not strip first lines, truncate at a numerical answer, or clean repeated text.',
    'primary_aggregation': 'For each of 27 unique source bundles, mean over its 7 or 8 request outputs; '
        'then unweighted mean of the 27 bundle means. Keep every request, including length-stopped and empty outputs.',
    'secondary_aggregation': 'Request-weighted mean over all 192 long requests; label repetition and unequal 7/8 multiplicity.',
    'paired_comparison': 'Same request identities/source bundle/order in each arm; report per-bundle paired differences '
        'and do not treat 192 repetitions as independent documents or use their count as an independent sample size.',
    'termination': 'Report EOS/stop, length/cap, errors, empty outputs, output token and word counts separately. '
        'Length-stopped output remains in primary score; natural EOS does not certify completeness.',
    'fixed_manual_audit': 'Source indices 0, 120, 194 (first/middle/last eligible). For each, inspect first request '
        'and first request of the opposite stop-versus-length type if present, using first Host arm selection; '
        'at most 6 identities, paired across arms. Check factual support, coverage of all source passages, '
        'repetition and visible truncation; preserve evidence; no overall quality certification.',
    'short_task': 'The 64 original GSM8K requests retain their existing frozen numeric extraction/scoring rule; '
        'report separately, never combine with ROUGE.',
    'limits': 'ROUGE overlap is not factual faithfulness; 27 bundles/72 distinct passage texts are not '
        'demonstrated statistically independent stories. No quality score or performance claim exists yet.',
    'scorer_dependency_status': 'Official source and definition frozen; local rouge dependency absent and '
        'not installed. Future scoring must record and validate its dependency version before reporting results.',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def freeze(path, value):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == data, f'Refusing to overwrite different frozen file: {path}'
    else:
        path.write_bytes(data)
    return sha(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizers-extension', type=Path,
                        help='Existing tokenizers native module, only if package import is unavailable.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / 'inputs_multinews_candidate'
    for name, expected in EXPECTED.items():
        assert sha((root / name).read_bytes()) == expected, f'Source changed: {name}'
    if args.tokenizers_extension:
        spec = importlib.util.spec_from_file_location('tokenizers', args.tokenizers_extension)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        import tokenizers as module
    assert module.__version__ == '0.23.2', module.__version__
    tok = module.Tokenizer.from_file(str(root / 'inputs_pro6000/sources/tokenizer.json'))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    raw = [json.loads(line) for line in (out / 'sources/multi_news.jsonl').read_text().splitlines()]
    template = json.loads((out / 'sources/official_dataset2prompt.json').read_text())['multi_news']
    selected, longs = [], []
    for index, source in enumerate(raw):
        # Preserve benchmark context verbatim, including passage labels and NEWLINE_CHAR markers.
        prompt = '<|endoftext|><|user|>\n' + template.format(**source) + '\n<|assistant|>\n'
        ids = encode(prompt)
        ref_lengths = [len(encode(answer)) for answer in source['answers']]
        if not (2500 <= len(ids) <= 3072 and ref_lengths and all(0 < n <= 1024 for n in ref_lengths)):
            continue
        selected.append({'source_index': index, 'source': source})
        longs.append(dict(task='multi_news', source_dataset='LongBench/multi_news',
            example_index=index, document_id=source['_id'], prompt=prompt,
            prompt_token_ids=ids, prompt_token_ids_sha256=token_sha(ids),
            source_context_sha256=sha(source['context'].encode()), gold=list(source['answers']),
            reference_answer=list(source['answers']), reference_token_counts=ref_lengths,
            quality_reference='Official fixed-revision LongBench multi_news answers, no invented summary.',
            output_tokens=1024, request_class='long', source_context_complete=True,
            synthetic_extension=False, source_sample_id=f'LongBench/multi_news:{index}'))
    assert [r['source_index'] for r in selected] == ELIGIBLE
    assert len({r['source_context_sha256'] for r in longs}) == len(longs) == 27
    previous = json.loads((root / 'inputs_pro6000/pro6000_high_256.json').read_text())
    shorts = [row for row in previous['source_requests'] if row['request_class'] == 'short']
    assert len(shorts) == 64
    rows, repetitions = [], Counter()
    for index in range(256):
        source = shorts[index // 4] if index % 4 == 0 else longs[(index - index // 4 - 1) % len(longs)]
        key = source['source_sample_id']
        row = dict(source, request_id=f'E-multinews256-{index:04d}', arrival_s=0.0,
                   repetition_index=repetitions[key], independent_request_identity=True,
                   cache_salt=f'E-measured-{index}', warmup_cache_salt=f'E-warm-{index}',
                   prompt_tokens=len(source['prompt_token_ids']))
        assert encode(row['prompt']) == row['prompt_token_ids']
        assert token_sha(row['prompt_token_ids']) == row['prompt_token_ids_sha256']
        assert row['prompt_tokens'] + row['output_tokens'] <= 4096
        repetitions[key] += 1
        rows.append(row)
    assert len({r['request_id'] for r in rows}) == len({r['cache_salt'] for r in rows}) == 256
    assert all(r['prompt_token_ids'] == s['prompt_token_ids'] and r['output_tokens'] == s['output_tokens']
               for r, s in zip(rows[::4], shorts))
    passages = [text.strip() for r in selected
                for text in re.split(r'(?m)^Passage \d+:\n', r['source']['context']) if text.strip()]
    # Splitting above is only for descriptive duplicate counting, never construction of model input.
    assert len(passages) == 73 and len(set(passages)) == 72
    stats = dict(requests=256, short_requests=64, long_requests=192,
        distinct_long_source_bundles=27, original_passage_count=73, distinct_passage_texts=72,
        unique_prompts=91, long_prompt_tokens_min=min(len(r['prompt_token_ids']) for r in longs),
        long_prompt_tokens_max=max(len(r['prompt_token_ids']) for r in longs),
        long_reference_tokens_min=min(n for r in longs for n in r['reference_token_counts']),
        long_reference_tokens_max=max(n for r in longs for n in r['reference_token_counts']),
        long_repetitions_by_source=dict(Counter(str(r['example_index']) for r in rows if r['request_class'] == 'long')),
        output_token_caps_total=sum(r['output_tokens'] for r in rows),
        prompt_tokens_total=sum(r['prompt_tokens'] for r in rows),
        max_prompt_plus_output_cap=max(r['prompt_tokens'] + r['output_tokens'] for r in rows),
        nominal_prompt_only_KV_gib=sum(((r['prompt_tokens'] + 15) // 16) * 16 * 131072 for r in rows) / 1024**3,
        nominal_complete_KV_gib=sum(((r['prompt_tokens'] + r['output_tokens'] + 15) // 16) * 16 * 131072
                                  for r in rows) / 1024**3,
        estimate_caveat='Hypothetical simultaneous residence, not measured GPU allocation or a trigger guarantee; '
            'natural output lengths and retirement can make actual peak much lower.')
    selected_sha = freeze(out / 'sources/multi_news_selected.json', selected)
    quality_sha = freeze(out / 'quality_contract.json', QUALITY)
    provenance = dict(verified_inputs=EXPECTED, generator_sha256=sha(Path(__file__).read_bytes()),
        tokenizer_version=module.__version__, source_revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672',
        selected_source_indices=ELIGIBLE, selected_raw_records_sha256=selected_sha,
        quality_contract_sha256=quality_sha,
        selection='All official records with complete prompt 2500–3072 tokens and nonempty references <=1024 tokens; '
            'fixed before observing any generated outputs.',
        gsm8k=previous['source_manifest']['gsm8k'])
    provenance['gsm8k']['path'] = 'inputs_pro6000/sources/gsm8k_test_selected.json'
    workload = dict(schema='E.pro6000.workload.v1', name='pro6000_multinews_candidate_256',
        status='CPU_FROZEN_GPU_UNRUN', research_scope=previous['research_scope'],
        source_manifest=provenance, source_requests=rows, statistics=stats,
        summary_instruction_template=template,
        workload_kind='repeated_complete_natural_multinews_reference_summary_candidate',
        repetition_disclosure='27 original full benchmark news bundles, first three repeated 8 times and others '
            '7 times: 192 long requests are not 192 independent documents. Benchmark-supplied multi-passage '
            'contexts are retained verbatim; no agent truncation, concatenation, padding or synthetic extension.',
        generation_semantics=dict(measured_natural_eos=True, required_run_cell_flag='--natural',
            temperature=0.0, min_tokens=0, ignore_eos=False,
            output_tokens='per-request cap, not forced length', gold_summary_available=True),
        arrival_trace='All arrivals at 0 seconds. One original short request then three long requests; '
            'long sources cycle through all 27 eligible indices in original dataset order.',
        cache_identity='Current run_cell uses E-{episode}-{index}; frozen salts match warm/measured episode names.',
        suggested_fixed_resources=previous['suggested_fixed_resources'],
        comparison_contract=previous['comparison_contract'],
        execution_boundary='No GPU submitted. Do not reuse E249/E223 target spec on different prompts; '
            'a future intervention needs its own observed legal event and prefrozen target. '
            'No changes to existing in-flight experiments or runtime parameters.')
    name = 'pro6000_multinews_candidate_256.json'
    workload_sha = freeze(out / name, workload)
    receipt = dict(status='CPU_FROZEN_GPU_UNRUN', file=name, sha256=workload_sha,
        generator_sha256=provenance['generator_sha256'], selected_raw_records_sha256=selected_sha,
        quality_contract_sha256=quality_sha, tokenizer_version=module.__version__,
        all_256_prompts_reencoded_exact=True, all_256_prompt_plus_cap_legal=True,
        original_64_short_prompts_and_caps_preserved=True, source_contexts_retained_verbatim=True,
        unique_identities_and_salts=True, statistics=stats)
    freeze(out / 'freeze_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
