#!/usr/bin/env python3
"""Select the next 32 source-ordered documents, excluding all three prior cohorts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from prepare_rotation_holdout import articles, file_sha, sha, token_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-arrow', required=True, type=Path)
    p.add_argument('--prior-inputs', required=True, nargs=3, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    priors, by_start = [], {}
    for root in args.prior_inputs:
        config, workload = [json.loads((root/n).read_text()) for n in ('config.json', 'workload.json')]
        assert sha(json.dumps(workload, sort_keys=True).encode()) == config['workload_sha256']
        assert (config['requests'], config['prompt_tokens'], config['output_tokens'], config['arrival_gap_s']) == (32,3072,1024,.05)
        assert len(workload['source_requests']) == len(workload['actual_prompt_token_ids']) == 32
        for row, ids in zip(workload['source_requests'], workload['actual_prompt_token_ids']):
            start = row['dataset_row_index']
            assert start not in by_start and len(ids) == 3072 and token_sha(ids) == row['prompt_token_ids_sha256']
            by_start[start] = (row, ids)
        priors.append(dict(root=str(root.resolve()), workload_sha256=config['workload_sha256'],
                           files_sha256={n:file_sha(root/n) for n in ('config.json','workload.json')}))
    assert len(by_start) == 96
    base = json.loads((args.prior_inputs[0]/'config.json').read_text())
    old = json.loads((args.prior_inputs[0]/'workload.json').read_text())
    source, model = base['source'], base['model']
    assert model['revision'] == model['tokenizer_revision'] == '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
    assert source['dataset_revision'] == 'b08601e04326c79dfdd32d625aee71d232d685c3'
    assert args.dataset_arrow.name == source['arrow_file'] and file_sha(args.dataset_arrow) == source['arrow_sha256']
    for name, expected in dict(source['tokenizer_files_sha256'], **{'config.json':source['model_config_sha256']}).items():
        path = Path(cached_file(model['id'], name, revision=model['revision'], local_files_only=True))
        assert file_sha(path) == expected
    tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], local_files_only=True)
    dataset = Dataset.from_file(str(args.dataset_arrow))
    excluded = [{k:r[k] for k in ('document_id','document_sha256','prompt_token_ids_sha256','dataset_row_index','dataset_row_end_exclusive')}
                for r, _ in by_start.values()]
    end = max(r['dataset_row_end_exclusive'] for r in excluded)
    used_docs = {r['document_sha256'] for r in excluded}
    used_prompts = {r['prompt_token_ids_sha256'] for r in excluded}
    assert len(used_docs) == len(used_prompts) == 96
    selected, reproduced, skips = [], set(), []
    for start, stop, title, document in articles(dataset):
        if start < end and start not in by_start:
            continue
        ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)['input_ids']
        digest = sha(document.encode())
        if start in by_start:
            row, prior_ids = by_start[start]
            assert stop == row['dataset_row_end_exclusive'] and document == row['prompt']
            assert digest == row['document_sha256'] and len(ids) == row['original_document_token_count'] and ids[:3072] == prior_ids
            reproduced.add(start)
            continue
        if len(ids) < 3072:
            continue
        prompt_hash = token_sha(ids[:3072])
        if digest in used_docs or prompt_hash in used_prompts:
            skips.append(dict(start=start, document_sha256=digest, prompt_token_ids_sha256=prompt_hash))
            continue
        used_docs.add(digest); used_prompts.add(prompt_hash)
        selected.append((start, stop, title, document, digest, ids, prompt_hash))
        if len(selected) == 32:
            break
    assert len(reproduced) == 96 and len(selected) == 32
    intervals = sorted([(r['dataset_row_index'],r['dataset_row_end_exclusive']) for r in excluded] + [(r[0],r[1]) for r in selected])
    assert all(a[1] <= b[0] for a,b in zip(intervals,intervals[1:]))
    records, prompts = [], []
    for start,stop,title,document,digest,ids,prompt_hash in selected:
        rid = f'memory-train-article-{start:07d}'
        records.append(dict(request_id=rid, sample_id=start, dataset_row_index=start,
            dataset_row_end_exclusive=stop, document_id=rid, article_title=title, document_sha256=digest,
            prompt=document, prompt_sha256=digest, prompt_token_count=3072,
            prompt_token_ids_sha256=prompt_hash, original_document_token_count=len(ids)))
        prompts.append(ids[:3072])
    scope = 'Document, row and prompt-prefix disjoint from original32 and holdout cohort0/1 (96 prior requests); same dataset, lengths and steady-arrival regime, not an independent population.'
    workload = dict(schema=old['schema'], source_requests=records, actual_prompt_token_ids=prompts,
        arrival_traces_s=deepcopy(old['arrival_traces_s']), arrival_rule=old['arrival_rule'], document_identity_scope=scope)
    config = deepcopy(base)
    config.update(source=dict(source, cohort_id='cohort2', identity_scope=scope,
        selection_rule='next32 complete articles with >=3072 tokens after all96 prior row intervals, in pinned shard source order; exclude duplicate document and prompt-prefix hashes',
        excluded_original_row_prefix_end_exclusive=end), workload_sha256=sha(json.dumps(workload,sort_keys=True).encode()),
        input_preparation='offline fresh32 prefix inputs; GPU UNRUN; inherited SLO fields are reference only')
    receipt = dict(schema_version=1, status='INPUTS_PREPARED_GPU_UNRUN', cohort_id='cohort2',
        workload_sha256=config['workload_sha256'], prior_inputs=priors, excluded_prior_requests=excluded,
        checks=dict(prior96_documents_and_tokens_reproduced=True, all128_document_hashes_unique=True,
                    all128_prompt_hashes_unique=True, source_intervals_disjoint=True, closed_article_boundaries=True),
        duplicate_skips=skips, script_sha256=file_sha(Path(__file__)), command=[sys.executable]+sys.argv,
        dataset_arrow=str(args.dataset_arrow.resolve()), dataset_arrow_sha256=file_sha(args.dataset_arrow), scope=scope)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name,value in [('config.json',config),('workload.json',workload),('inputs_report.json',receipt)]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],workload_sha256=config['workload_sha256'],
        first_row=records[0]['dataset_row_index'],last_row_end_exclusive=records[-1]['dataset_row_end_exclusive'],checks=receipt['checks'])))


if __name__ == '__main__':
    main()
