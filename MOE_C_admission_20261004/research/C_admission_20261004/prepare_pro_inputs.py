"""Prepare disjoint 192/192 article cohorts from three existing tokenized inputs."""
import argparse
import hashlib
import json
from pathlib import Path


RELATIVE_ROOT = Path('refine-logs/expert_saturation/experiments/admission_capacity/'
                     '20260929_commit_recheck')
INPUTS = (
    ('A', 'spare_followup_fresh_inputs_r01',
     'd336841dbaf24bd75d1bc8eb16f6f31aa9171325b7a537ecdf80fed639f7f730'),
    ('B', 'native_residency_fresh_inputs_r01',
     '4529a6f680ec0e67ca2082078e583786f8115ea99b101d9a5dc0ba00e7c66e80'),
    ('C', 'native_current_guard_fresh_inputs_r01',
     'f6e5d288fa54012851be92cacb77480e4d6601e92d53315e65c8c5d04f7b9974'),
)
REVISION = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def describe(lengths):
    return dict(requests=len(lengths), minimum=min(lengths), maximum=max(lengths),
                mean=sum(lengths) / len(lengths), total=sum(lengths),
                short_lt1536=sum(n < 1536 for n in lengths),
                medium_1536_2559=sum(1536 <= n < 2560 for n in lengths),
                long_ge2560=sum(n >= 2560 for n in lengths),
                prompt_rounded_pages=sum((n + 15) // 16 for n in lengths),
                output_limit_rounded_pages=sum((n + 1024 + 15) // 16 for n in lengths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('pro_inputs.json'))
    args = parser.parse_args()
    if not args.repository.is_absolute():
        parser.error('--repository must be the original checkout absolute path')
    repository = args.repository.resolve()
    sources, manifests, skipped = [], [], []
    seen_ids, seen_docs, seen_tokens = set(), set(), set()
    shared_model = shared_dataset = None
    for label, name, expected_sha in INPUTS:
        relative = RELATIVE_ROOT / name / 'workload.json'
        path = repository / relative
        payload = path.read_bytes()
        if sha(payload) != expected_sha:
            raise ValueError(f'Frozen input changed: {path}')
        workload = json.loads(payload)
        config_path = path.with_name('config.json')
        config_payload = config_path.read_bytes()
        config = json.loads(config_payload)
        model = config['model']
        if (model['id'] != 'allenai/OLMoE-1B-7B-0924' or model['dtype'] != 'bfloat16'
                or model['revision'] != REVISION or model['tokenizer_revision'] != REVISION):
            raise ValueError(f'Model/tokenizer differs: {path}')
        dataset = {key: config['source'][key] for key in
                   ('dataset_id', 'dataset_config', 'split', 'dataset_revision',
                    'parquet_sha256', 'tokenizer_files_sha256')}
        if shared_model is None:
            shared_model, shared_dataset = model, dataset
        if model != shared_model or dataset != shared_dataset:
            raise ValueError('Source model/tokenizer/dataset identities differ')
        if len(workload['source_requests']) != len(workload['actual_prompt_token_ids']):
            raise ValueError('Token/source arrays are misaligned')
        records = []
        for index, (record, tokens) in enumerate(zip(workload['source_requests'],
                                                    workload['actual_prompt_token_ids'])):
            token_hash = sha(json.dumps(tokens, separators=(',', ':')).encode())
            if token_hash != record['prompt_token_ids_sha256']:
                raise ValueError('Per-request token hash differs')
            if (len(tokens) != record['prompt_token_count']
                    or len(tokens) != record['original_document_token_count']
                    or not 1 <= len(tokens) <= 3072 or len(tokens) + 1024 > 4096):
                raise ValueError('Full prompt/context contract differs')
            if (sha(record['prompt'].encode()) != record['prompt_sha256']
                    or record['prompt_sha256'] != record['document_sha256']):
                raise ValueError('Full source text hash differs')
            document_id, document_hash = record['document_id'], record['document_sha256']
            if (document_id in seen_ids or document_hash in seen_docs
                    or token_hash in seen_tokens):
                skipped.append(dict(source=label, original_source_index=index,
                                    document_id=document_id, token_hash=token_hash))
                continue
            seen_ids.add(document_id)
            seen_docs.add(document_hash)
            seen_tokens.add(token_hash)
            # Token payloads and source identities are sufficient for the runner;
            # do not duplicate long article text in this compact artifact.
            copied = {key: value for key, value in record.items() if key != 'prompt'}
            copied.update(original_request_id=record['request_id'], source_input=label,
                          original_source_index=index, source_workload=str(relative))
            records.append((copied, tokens))
        if len(records) != 128:
            raise ValueError(f'Expected 128 unique frozen articles from {label}')
        sources.append(records)
        manifests.append(dict(label=label, workload=str(relative), workload_sha256=sha(payload),
                              config=str(config_path.relative_to(repository)),
                              config_sha256=sha(config_payload), unique_requests=len(records)))
    result = {}
    for split, parity in (('dev', 0), ('test', 1)):
        ordered = [source[index] for index in range(parity, 128, 2) for source in sources]
        requests, token_lists = [], []
        for index, (record, tokens) in enumerate(ordered):
            requests.append(dict(record, source_index=index))
            token_lists.append(tokens)
        result[split] = dict(
            schema='olmoe-natural-cadence-holdout-v1', source_requests=requests,
            actual_prompt_token_ids=token_lists,
            arrival_traces_s={'steady': [0.] * len(requests)},
            arrival_rule='Default all-at-zero placeholder; the runner freezes each low/near/high '
                         'external-arrival trace identically across policy arms.',
            output_contract=dict(max_output_tokens=1024, ignore_eos=False, min_tokens=0),
            input_stats=describe(list(map(len, token_lists))))
    if {r['prompt_token_ids_sha256'] for r in result['dev']['source_requests']} & {
            r['prompt_token_ids_sha256'] for r in result['test']['source_requests']}:
        raise ValueError('Development/test prompt overlap')
    result['provenance'] = dict(
        repository=str(repository), sources=manifests, model=shared_model, dataset=shared_dataset,
        preparation='CPU-only; no network, tokenization, output/EOS observations or GPU use.',
        split_rule='Within each 128-record source, even original indices go to dev and odd '
                   'indices to test. Interleave A/B/C at each retained source index, giving '
                   '64 articles from each source per 192-request split.',
        deduplication='Keep first occurrence in A/B/C order by document ID, full-text SHA256 '
                      'or canonical full-token-array SHA256; validate all per-request hashes.',
        duplicates_removed=skipped, unique_documents=384,
        independence='Dev/test are document/text/full-token disjoint for C development. '
                     'All source packages have prior experiment use; not globally unseen or '
                     'strict blind test evidence.',
        resources=dict(dtype='bfloat16', max_model_len=4096, output_limit=1024,
                       engine_max_num_seqs=256, token_bytes=131072, block_size=16,
                       block_bytes=2097152, usable_kv_pages=32768, null_pages=1,
                       total_kv_pages=32769, kv_cache_memory_bytes=64 * 1024**3 + 2097152),
        development_caps=[96, 128, 192], native_cap=256,
        proposed_pressure_points=[dict(name='low', requests=64, arrival_gap_s=.5),
                                  dict(name='near', requests=192, arrival_gap_s=.1),
                                  dict(name='high', requests=192, arrival_gap_s=.02)],
        pressure_status='UNRUN; inspect actual KV free, active and true recovery backlog '
                        'before assigning an observed pressure category.')
    with args.output.open('x') as stream:
        json.dump(result, stream, separators=(',', ':'), ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size,
                          unique_documents=384, duplicates_removed=len(skipped),
                          dev=result['dev']['input_stats'], test=result['test']['input_stats']),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
