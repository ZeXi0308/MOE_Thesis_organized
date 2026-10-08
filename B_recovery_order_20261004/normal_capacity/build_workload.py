#!/usr/bin/env python3
"""CPU-only unique natural-text prefixes for normal-memory concurrency characterization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def token_sha(ids):
    return digest(json.dumps(ids, separators=(',', ':')).encode())


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--assets', type=Path, default=Path('/private/tmp/moe-a-fresh-input-assets-20261001'))
    parser.add_argument('--repo', type=Path, default=Path('/Users/zhaozhenyu/Desktop/毕业设计/MOE_Thesis_organized'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); destination = args.output or args.base/'normal_capacity'/'inputs'
    if destination.exists():
        raise FileExistsError(destination)
    base_work = json.loads((args.base/'pkg/inputs/workload.json').read_text())
    base_config = json.loads((args.base/'pkg/inputs/config.json').read_text())
    start_row = max(r['dataset_row_end_exclusive'] for r in base_work['source_requests'])
    expected = dict(base_config['source']['tokenizer_files_sha256'],
                    **{'config.json': '3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335'})
    for name, sha in expected.items():
        assert file_sha(args.assets/name) == sha, name
    parquet = args.assets/'train-00000-of-00002.parquet'
    assert file_sha(parquet) == base_config['source']['parquet_sha256']
    model_config = json.loads((args.assets/'config.json').read_text())
    assert model_config['max_position_embeddings'] == 4096
    bytes_per_token = 2 * model_config['num_hidden_layers'] * model_config['num_key_value_heads'] * (model_config['hidden_size']//model_config['num_attention_heads']) * 2
    assert bytes_per_token == 131072
    helper_dir = args.repo/'refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck'
    sys.path.insert(0, str(helper_dir))
    from build_native_residency_fresh_inputs_r01 import ParquetRows, closed_articles
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.assets/'tokenizer.json'))
    selected = {'short': [], 'long': []}; seen_documents = set(); seen_tokens = set(); scanned = 0
    for begin, end, text in closed_articles(ParquetRows(parquet)):
        if begin < start_row:
            continue
        scanned += 1; ids = tokenizer.encode(text, add_special_tokens=True).ids
        category = 'long' if len(ids) >= 3072 else 'short' if len(ids) >= 512 else None
        if category is None or len(selected[category]) >= 128:
            continue
        count = 3072 if category == 'long' else 512; prompt_ids = ids[:count]
        document_hash = digest(text.encode()); ids_hash = token_sha(prompt_ids)
        if document_hash in seen_documents or ids_hash in seen_tokens:
            continue
        seen_documents.add(document_hash); seen_tokens.add(ids_hash)
        document_id = f'memory-train-article-{begin:07d}'; prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        row = dict(request_id=f'b-normal-{begin:07d}-{category}', document_id=document_id,
                   dataset_row_index=begin, dataset_row_end_exclusive=end, article_title=text.splitlines()[0].strip(),
                   document_sha256=document_hash, prompt=prompt_text, prompt_sha256=digest(prompt_text.encode()),
                   prompt_token_ids_sha256=ids_hash, prompt_token_count=count, original_document_token_count=len(ids),
                   prefix_tokens=count, truncated=len(ids)>count, category=category, max_output_tokens=1024,
                   text_semantics='Decoded first N source tokens for display; actual_prompt_token_ids is authoritative; no concatenation, repetition, or padding')
        selected[category].append((row, prompt_ids))
        if all(len(v)==128 for v in selected.values()):
            break
    assert all(len(v)==128 for v in selected.values()), {k:len(v) for k,v in selected.items()}
    pairs = [entry for i in range(128) for entry in (selected['long'][i], selected['short'][i])]
    rows, tokens = zip(*pairs)
    for i, row in enumerate(rows):
        row['source_index'] = i
    workload = dict(schema='b-normal-capacity-natural-prefix-v1', source_requests=rows,
                    actual_prompt_token_ids=tokens, arrival_traces_s={'steady': [i*.1 for i in range(256)]},
                    arrival_rule='256 unique article prefixes, alternating long/short; fixed external 10 requests/s for 25.5s; independent of runtime completions',
                    output_contract=dict(max_output_tokens=1024, ignore_eos=False, min_tokens=0, actual_output_lengths='UNKNOWN_UNTIL_EXECUTION'))
    config = dict(status='CPU_PREPARED_GPU_UNRUN', model=base_config['model'], requests=256,
                  prompt_tokens=3072, prompt_tokens_by_request=[len(t) for t in tokens], output_tokens=1024,
                  max_output_tokens=1024, output_tokens_by_request={}, ignore_eos=False, min_tokens=0,
                  cap=256, engine_max_num_seqs=256, max_num_batched_tokens=1024, max_model_len=4096,
                  gpu_memory_utilization=.90, gpu_budget_semantics='Profile native .90 budget once, then freeze actual bytes across cells; no artificial small KV cap',
                  offload_gib=16, selective_save='on', enable_prefix_caching=False, seed=base_config['seed'],
                  arrival_gap_s=.1, arrival_span_s=25.5, arrival_process='alternating_natural_prefix_10rps',
                  workload_sha256=digest(json.dumps(workload, sort_keys=True).encode()),
                  source=dict(base_config['source'], selection=f'First128 unique articles in each length group after row{start_row}; short512..3071=>512prefix; long>=3072=>3072prefix; alternate long/short',
                              prefix_truncation=True, no_task_quality_claim=True))
    variants = []
    for cap in (64, 192, 256):
        balanced_prompt = cap//2*(3072+512); balanced_end = cap//2*(4096+1536)
        nlong = min(cap,128); nshort = cap-nlong
        variants.append(dict(cap=cap, engine_max_num_seqs=256, requests=256,
                             balanced_prompt_tokens=balanced_prompt, balanced_endpoint_tokens=balanced_end,
                             balanced_prompt_gib=balanced_prompt*bytes_per_token/2**30,
                             balanced_endpoint_gib=balanced_end*bytes_per_token/2**30,
                             maximum_any_active_subset_endpoint_tokens=nlong*4096+nshort*1536,
                             maximum_any_active_subset_endpoint_gib=(nlong*4096+nshort*1536)*bytes_per_token/2**30))
    design = dict(status='CPU_PREPARED_GPU_UNRUN', source_start_row=start_row, articles_examined=scanned,
                  unique_documents=len(seen_documents), unique_prompt_hashes=len(seen_tokens), short=128, long=128,
                  prompt_lengths=[512,3072], output_cap=1024, eos_enabled=True, max_model_len=4096,
                  kv_bytes_per_token=bytes_per_token, block_tokens=16, total_prompt_tokens=sum(map(len,tokens)),
                  total_endpoint_tokens=sum(len(t)+1024 for t in tokens), total_prompt_gib=56., total_endpoint_gib=88.,
                  variants=variants, fixed=dict(arrival_rps=10., arrival_span_s=25.5, max_num_batched_tokens=1024, engine_max_num_seqs=256),
                  selection_independence='No GPU outcomes used; inputs differ from prior B measured and warmup source rows, not a global unseen-data claim',
                  workload_scope='Natural article-prefix continuations with natural EOS; controlled length mix and arrivals, not a complete natural task or quality benchmark',
                  bounds='Endpoint bounds assume requests survive to cap; EOS, short-request departures and prefill scheduling determine actual residency. Cap192 can approach70GiB only with long-request enrichment. Do not claim pressure before measuring actual allocation and preemptions.',
                  next='Run native characterization at caps64/192/256 using same unique inputs and fixed profiled normal GPU KV bytes; at most one observed natural queueing point advances to native/flush_first reverse comparison',
                  provenance=dict(builder_sha256=file_sha(Path(__file__)), parquet_sha256=file_sha(parquet), tokenizer_sha256=expected['tokenizer.json'], base_workload_sha256=file_sha(args.base/'pkg/inputs/workload.json')))
    destination.mkdir(parents=True)
    write(destination/'workload.json', workload); write(destination/'config.json', config); write(destination/'design.json', design)
    for cap in (64, 192, 256):
        child = destination/f'cap{cap}'; child.mkdir()
        write(child/'workload.json', workload); write(child/'config.json', dict(config, cap=cap))
    print(json.dumps(dict(output=str(destination), variants=variants, unique_prompts=len(seen_tokens), articles_examined=scanned), indent=2))


if __name__ == '__main__':
    main()
