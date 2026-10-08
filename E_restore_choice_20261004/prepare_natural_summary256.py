"""Freeze one complete-document natural-output workload; CPU tokenizers only.

Requires E's existing inputs_pro6000 sources. Does not import torch, load model
weights, change a scheduler, or start a benchmark. Existing different files are
never overwritten. Run with /root/miniconda3/bin/python on the authorized host.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'inputs_pro6000/pro6000_high_256.json': '6b2a65dc10ca32cc1d8b84e9c7c83e90796cb5e8eed1f1f0e0d98301b431c88b',
    'inputs_pro6000/sources/multifieldqa_en_selected.json': '3a3ee3cfafe04ca1afff3b82e6b4da9679402a941fe350bcaf2f1fd29fb59de9',
    'inputs_pro6000/sources/tokenizer.json': 'b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167',
}
INSTRUCTION = ('Produce a detailed, faithful briefing of the document below for a reader who has not seen it. '
    'Use 6–8 titled sections, covering the principal people or entities, chronology, concrete facts, '
    'supporting evidence, and any limitations. Aim for 600–750 words. Do not invent missing details.')
SELECTED = {7: 3052, 15: 2871, 62: 2736}
WORKLOAD_FILE = 'pro6000_natural_summary_256.json'


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def freeze(path, value):
    data = encoded(value)
    if path.exists():
        assert path.read_bytes() == data, f'Refusing to overwrite a different freeze: {path}'
    else:
        with path.open('xb') as f:
            f.write(data)
    return sha(data)


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=root)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--model-tokenizer', type=Path, required=True,
                   help='Actual existing model tokenizer.json, checked against frozen input tokenizer.')
    a = p.parse_args()
    root = a.root
    out = a.out or root/'inputs_natural_summary256'
    for relative, expected in EXPECTED.items():
        assert sha((root/relative).read_bytes()) == expected, f'Source changed: {relative}'
    assert sha(a.model_tokenizer.read_bytes()) == EXPECTED['inputs_pro6000/sources/tokenizer.json']
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.model_tokenizer))
    encode = lambda prompt: tok.encode(prompt, add_special_tokens=False).ids
    previous = json.loads((root/'inputs_pro6000/pro6000_high_256.json').read_text())
    short = [r for r in previous['source_requests'] if r['request_class'] == 'short']
    assert len(short) == 64
    documents = {r['source_index']: r['source'] for r in json.loads(
        (root/'inputs_pro6000/sources/multifieldqa_en_selected.json').read_text())}
    long = []
    for index, expected_length in SELECTED.items():
        source = documents[index]
        prompt = '<|endoftext|><|user|>\n' + INSTRUCTION + '\n\nDocument:\n' + source['context'] + '\n<|assistant|>\n'
        ids = encode(prompt)
        assert len(ids) == expected_length and len(ids) + 1024 <= 4096
        long.append(dict(task='document_summary', source_dataset='multifieldqa_en',
            example_index=index, document_id=source['_id'], prompt=prompt,
            prompt_token_ids=ids, prompt_token_ids_sha256=token_sha(ids),
            source_context_sha256=sha(source['context'].encode()),
            gold=None, reference_answer=None, quality_reference='original full source document; no gold summary',
            output_tokens=1024, request_class='long', source_context_complete=True,
            synthetic_extension=False, source_sample_id=f'multifieldqa_en_summary:{index}'))
    rows = []
    repetitions = Counter()
    for index in range(256):
        source = short[index//4] if index % 4 == 0 else long[(index-index//4-1) % 3]
        key = source['source_sample_id']
        row = dict(source, request_id=f'E-natural-summary256-{index:04d}',
            arrival_s=0.0, repetition_index=repetitions[key], independent_request_identity=True,
            cache_salt=f'E-measured-{index}', warmup_cache_salt=f'E-warm-{index}',
            prompt_tokens=len(source['prompt_token_ids']))
        assert encode(row['prompt']) == row['prompt_token_ids']
        assert row['prompt_token_ids_sha256'] == token_sha(row['prompt_token_ids'])
        assert row['prompt_tokens'] + row['output_tokens'] <= 4096
        repetitions[key] += 1
        rows.append(row)
    assert len({r['request_id'] for r in rows}) == len({r['cache_salt'] for r in rows}) == 256
    assert Counter(r['example_index'] for r in rows if r['request_class'] == 'long') == {7:64, 15:64, 62:64}
    assert all(r['prompt_token_ids'] == s['prompt_token_ids'] and r['output_tokens'] == s['output_tokens']
               for r, s in zip(rows[::4], short))
    stats = dict(requests=256, short_requests=64, long_requests=192, unique_prompts=67,
        distinct_long_source_articles=3, long_prompt_tokens_by_source=SELECTED,
        output_token_caps_total=sum(r['output_tokens'] for r in rows),
        prompt_tokens_total=sum(r['prompt_tokens'] for r in rows),
        max_prompt_plus_output_cap=max(r['prompt_tokens']+r['output_tokens'] for r in rows),
        nominal_complete_KV_gib=sum(((r['prompt_tokens']+r['output_tokens']+15)//16)*16*131072
                                  for r in rows)/1024**3,
        estimate_caveat='Joint residence at all output caps is an upper-bound pressure estimate, not observed allocation.')
    assert stats['nominal_complete_KV_gib'] == 99.2421875
    provenance = dict(verified_inputs=EXPECTED, generator_sha256=sha(Path(__file__).read_bytes()),
        original_dataset_provenance={key: previous['source_manifest'][key] for key in ('gsm8k','longqa')},
        actual_model_tokenizer_sha256=sha(a.model_tokenizer.read_bytes()),
        full_context_source_indices=list(SELECTED), original_gsm8k_indices=list(range(128,192)))
    # Paths identify E-owned artifacts consistently across local and remote copies.
    for key, filename in [('gsm8k','gsm8k_test_selected.json'), ('longqa','multifieldqa_en_selected.json')]:
        provenance['original_dataset_provenance'][key]['path'] = 'inputs_pro6000/sources/'+filename
    workload = dict(schema='E.pro6000.workload.v1', name='pro6000_natural_summary_256',
        research_scope=previous['research_scope'], source_manifest=provenance,
        source_requests=rows, statistics=stats, summary_instruction=INSTRUCTION,
        workload_kind='repeated_complete_natural_document_summary_capacity_probe',
        repetition_disclosure='Three original full articles, each requested 64 times with independent identity; '
            'not 192 independent documents. No context truncation, concatenation or padding.',
        generation_semantics=dict(measured_natural_eos=True, required_run_cell_flag='--natural',
            temperature=0.0, min_tokens=0, ignore_eos=False, output_tokens='per-request cap, not forced length',
            gold_summary_available=False),
        arrival_trace='All arrivals are 0 seconds; fixed repeating short,long7,long15,long62 order.',
        cache_identity='Current run_cell uses E-{episode}-{index}; frozen salts match warm/measured episode names.',
        suggested_fixed_resources=previous['suggested_fixed_resources'],
        comparison_contract=previous['comparison_contract'])
    out.mkdir(parents=True, exist_ok=True)
    workload_sha = freeze(out/WORKLOAD_FILE, workload)
    receipt = dict(status='FROZEN_CPU_ONLY', file=WORKLOAD_FILE, sha256=workload_sha,
        generator_sha256=provenance['generator_sha256'], tokenizer_matches_actual_model=True,
        all_256_prompts_reencoded_exact=True, all_256_prompt_plus_cap_legal=True,
        original_64_short_prompts_and_caps_preserved=True, unique_identities_and_salts=True,
        natural_eos_required_for_measurement=True, statistics=stats)
    freeze(out/'freeze_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
