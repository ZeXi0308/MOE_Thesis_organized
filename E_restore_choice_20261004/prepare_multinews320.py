"""Freeze the sole conditional MultiNews320 candidate, CPU only; never queues a run."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from prepare_multinews_candidate import freeze, sha, token_sha


LOCKED = {
    'inputs_multinews_candidate/pro6000_multinews_candidate_256.json':
        '00fa8875af4fa3fde61d31dff503fda9216f98839ed57d9db58ac2480a1c30c6',
    'inputs_multinews_candidate/quality_contract.json':
        '65b6b3a4c52cb2bc0f6b29b5b43858db001f6ec47bc0de0a265166c25da7e101',
    'inputs_multinews_candidate/sources/multi_news_selected.json':
        'c6f49f4d97ffc9aa5f1ff2c5d50bffdf31ba7fb12cf7256b4124f1574326782c',
    'inputs_pro6000/sources/tokenizer.json':
        'b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167',
    'prepare_multinews_candidate.py':
        'c95046aee0494bc229243ccd368022454e85bcbae73688dfa5eabe2584a4384a',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizers-extension', type=Path, required=True,
                        help='Existing local tokenizers0.23.2 native extension; no installation.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name, expected in LOCKED.items():
        assert sha((root/name).read_bytes()) == expected, f'Frozen dependency changed: {name}'
    spec = importlib.util.spec_from_file_location('tokenizers', args.tokenizers_extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__version__ == '0.23.2'
    tok = module.Tokenizer.from_file(str(root/'inputs_pro6000/sources/tokenizer.json'))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    parent_path = 'inputs_multinews_candidate/pro6000_multinews_candidate_256.json'
    parent = json.loads((root/parent_path).read_text())
    original = parent['source_requests']
    assert len(original) == 256
    sources = {r['source_index']: r['source'] for r in json.loads(
        (root/'inputs_multinews_candidate/sources/multi_news_selected.json').read_text())}
    ordered = parent['source_manifest']['selected_source_indices']
    long_examples = {r['example_index']: r for r in original if r['request_class'] == 'long'}
    assert len(ordered) == len(long_examples) == 27
    rows = deepcopy(original)
    repetitions = Counter(r['source_sample_id'] for r in rows)
    for added_index in range(64):
        index = 256 + added_index
        source_index = ordered[(192 + added_index) % len(ordered)]
        source = long_examples[source_index]
        row = dict(source, request_id=f'E-multinews320-{index:04d}', arrival_s=0.0,
            repetition_index=repetitions[source['source_sample_id']],
            cache_salt=f'E-measured-{index}', warmup_cache_salt=f'E-warm-{index}')
        repetitions[source['source_sample_id']] += 1
        rows.append(row)
    assert rows[:256] == original
    assert len(rows) == len({r['request_id'] for r in rows}) == len({r['cache_salt'] for r in rows}) == 320
    assert Counter(r['request_class'] for r in rows) == {'short': 64, 'long': 256}
    longs = [r for r in rows if r['request_class'] == 'long']
    assert [r['example_index'] for r in longs] == [ordered[n % 27] for n in range(256)]
    for row in rows:
        assert encode(row['prompt']) == row['prompt_token_ids']
        assert len(row['prompt_token_ids']) == row['prompt_tokens']
        assert token_sha(row['prompt_token_ids']) == row['prompt_token_ids_sha256']
        assert row['prompt_tokens'] + row['output_tokens'] <= 4096
        if row['request_class'] == 'long':
            source = sources[row['example_index']]
            prompt = '<|endoftext|><|user|>\n' + parent['summary_instruction_template'].format(**source) + '\n<|assistant|>\n'
            assert row['prompt'] == prompt
            assert row['gold'] == row['reference_answer'] == source['answers']
    def geometry(records):
        return dict(requests=len(records), prompt_tokens=sum(r['prompt_tokens'] for r in records),
            output_caps=sum(r['output_tokens'] for r in records),
            prompt_blocks=sum((r['prompt_tokens'] + 15)//16 for r in records),
            cap_blocks=sum((r['prompt_tokens'] + r['output_tokens'] + 15)//16 for r in records),
            nominal_prompt_only_KV_gib=sum((r['prompt_tokens'] + 15)//16 for r in records)/512,
            nominal_complete_KV_gib=sum((r['prompt_tokens'] + r['output_tokens'] + 15)//16 for r in records)/512)
    workload = deepcopy(parent)
    stats = workload['statistics']
    full, appended = geometry(rows), geometry(rows[256:])
    stats.update(requests=320, short_requests=64, long_requests=256,
        long_repetitions_by_source=dict(Counter(str(r['example_index']) for r in longs)),
        output_token_caps_total=full['output_caps'], prompt_tokens_total=full['prompt_tokens'],
        nominal_prompt_only_KV_gib=full['nominal_prompt_only_KV_gib'],
        nominal_complete_KV_gib=full['nominal_complete_KV_gib'],
        prompt_blocks_total=full['prompt_blocks'], complete_cap_blocks_total=full['cap_blocks'],
        appended_long_geometry=appended)
    assert sorted(Counter(stats['long_repetitions_by_source'].values()).items()) == [(9, 14), (10, 13)]
    manifest = workload['source_manifest']
    manifest.update(parent_workload=parent_path, parent_workload_sha256=LOCKED[parent_path],
        parent_workload_generator_sha256=parent['source_manifest']['generator_sha256'],
        preserved_parent_prefix_requests=256, generator_sha256=sha(Path(__file__).read_bytes()),
        generation_dependencies_sha256=LOCKED)
    workload.update(name='pro6000_multinews_candidate_320', status='CPU_FROZEN_GPU_UNRUN',
        queue_status='NOT_QUEUED', execution_condition='CONDITIONAL_AFTER_SUMMARY256_EIGHT_ARM_RESULTS',
        source_requests=rows,
        repetition_disclosure='27 complete official news bundles; first 13 in fixed source order repeat 10 times, '
            'remaining 14 repeat 9 times. 256 long requests are not 256 independent documents. '
            'No prompt truncation, concatenation, padding, new download or forced output length.',
        arrival_trace='All arrivals at 0 seconds. The original 256 requests remain an exact prefix. '
            'Append 64 long requests, continuing the original 27-source cycle at long ordinal192.',
        execution_boundary='CPU preparation only, NOT_QUEUED. Conditional on completed review of the original '
            'summary256 H/R/L/B/B/L/R/H group; do not start or alter that group. No reused event target spec.',
        quality_scope_note='Original quality_contract.json remains unchanged. Before scoring this candidate, '
            'make an explicit scale-only scope addendum and thin input/count binding adaptation: '
            '320 total/256 long, per-bundle repetition9/10. Preserve ROUGE formula, 27-bundle macro averaging, '
            'errors, fixed sources0/120/194 and at-most-six-identity manual audit. Current256 CLI must reject this input.')
    workload['suggested_fixed_resources']['max_num_seqs'] = 320
    out = root/'inputs_multinews320'
    out.mkdir(exist_ok=True)
    name = 'pro6000_multinews_candidate_320.json'
    value = freeze(out/name, workload)
    receipt = dict(status='CPU_FROZEN_GPU_UNRUN', queue_status='NOT_QUEUED',
        execution_condition=workload['execution_condition'], file=name, sha256=value,
        generator_sha256=manifest['generator_sha256'], parent_sha256=LOCKED[parent_path],
        original_quality_contract_sha256=LOCKED['inputs_multinews_candidate/quality_contract.json'],
        tokenizer_version=module.__version__, all_320_prompts_reencoded_exact=True,
        original_256_request_records_preserved_exactly=True, all_320_prompt_plus_cap_legal=True,
        source_contexts_and_gold_verbatim=True, full_geometry=full, appended_long_geometry=appended,
        long_repetitions_by_source=stats['long_repetitions_by_source'],
        suggested_fixed_resources=workload['suggested_fixed_resources'])
    freeze(out/'freeze_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
