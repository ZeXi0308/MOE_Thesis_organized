#!/usr/bin/env python3
"""Prepare two disjoint WikiText cohorts offline; no runtime or GPU measurements."""
import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def articles(dataset):
    """Same title boundary and exact row concatenation as the original preparation."""
    start, pieces = None, []
    for index, row in enumerate(dataset):
        text = str(row['text'])
        if re.fullmatch(r'= [^=].*? =', text.strip()):
            if start is not None:
                yield start, index, pieces[0].strip(), ''.join(pieces)
            start, pieces = index, []
        if start is not None:
            pieces.append(text)
    # A shard's final article has no closing next-title boundary and is excluded.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-arrow', required=True, type=Path)
    parser.add_argument('--old-inputs', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    base = json.loads((args.old_inputs / 'config.json').read_text())
    old = json.loads((args.old_inputs / 'workload.json').read_text())
    source, model = base['source'], base['model']
    assert (base['requests'], base['prompt_tokens'], base['output_tokens']) == (32, 3072, 1024)
    assert base['arrival_gap_s'] == .05
    assert model['revision'] == model['tokenizer_revision'] == '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
    assert source['dataset_revision'] == 'b08601e04326c79dfdd32d625aee71d232d685c3'
    assert args.dataset_arrow.name == source['arrow_file']
    assert file_sha(args.dataset_arrow) == source['arrow_sha256']
    local_model = {}
    for name, expected in dict(source['tokenizer_files_sha256'], **{'config.json': source['model_config_sha256']}).items():
        path = Path(cached_file(model['id'], name, revision=model['revision'], local_files_only=True))
        assert file_sha(path) == expected, name
        local_model[name] = str(path)
    tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'], local_files_only=True)
    dataset = Dataset.from_file(str(args.dataset_arrow))
    old_records = old['source_requests']
    old_by_start = {record['dataset_row_index']: (record, ids) for record, ids in
                    zip(old_records, old['actual_prompt_token_ids'])}
    assert len(old_by_start) == len(old_records) == len(old['actual_prompt_token_ids']) == 32
    assert sha(json.dumps(old, sort_keys=True).encode()) == base['workload_sha256']
    exclusion_end = max(record['dataset_row_end_exclusive'] for record in old_records)
    used_hashes = {record['document_sha256'] for record in old_records}
    assert len(used_hashes) == 32
    selected, reproduced, duplicate_skips = [], set(), []
    for start, stop, title, document in articles(dataset):
        if start < exclusion_end and start not in old_by_start:
            continue
        digest = sha(document.encode())
        ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)['input_ids']
        if start in old_by_start:
            record, prior_ids = old_by_start[start]
            assert stop == record['dataset_row_end_exclusive']
            assert document == record['prompt'] and digest == record['document_sha256']
            assert len(ids) == record['original_document_token_count']
            assert ids[:3072] == prior_ids and token_sha(prior_ids) == record['prompt_token_ids_sha256']
            reproduced.add(start)
            continue
        assert start >= exclusion_end
        if len(ids) < 3072:
            continue
        if digest in used_hashes:
            duplicate_skips.append({'start': start, 'stop': stop, 'document_sha256': digest})
            continue
        used_hashes.add(digest)
        selected.append(dict(start=start, stop=stop, title=title, document=document,
                             document_sha256=digest, ids=ids))
        if len(selected) == 64:
            break
    assert len(reproduced) == 32 and len(selected) == 64
    intervals = sorted((record['dataset_row_index'], record['dataset_row_end_exclusive']) for record in old_records)
    intervals += [(article['start'], article['stop']) for article in selected]
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    assert len({token_sha(ids) for ids in old['actual_prompt_token_ids']} |
               {token_sha(article['ids'][:3072]) for article in selected}) == 96

    provenance = deepcopy(source)
    provenance.update(
        selection_rule='next 64 unique complete articles with >=3072 tokens after all original cohort rows, '
                       'in pinned train shard 0 source order; first 32 cohort0, next 32 cohort1',
        excluded_original_row_prefix_end_exclusive=exclusion_end,
        excluded_original_workload_sha256=base['workload_sha256'],
        observed_dataset_fingerprint=str(dataset._fingerprint),
        identity_scope='document- and row-disjoint from the original 32-request cohort and between these two cohorts; '
                       'same dataset, model, lengths and steady arrival regime; no population-independence claim',
        pre_run_length_rationale='same 3072 prompt +1024 output as the frozen rotation four-arm pilot; '
                                 'new article identities test transfer without changing lengths or arrival timing')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files, cohorts = {}, {}
    for cohort in range(2):
        label = f'cohort{cohort}'
        records, prompts = [], []
        for article in selected[cohort * 32:(cohort + 1) * 32]:
            ids = article['ids'][:3072]
            rid = f"memory-train-article-{article['start']:07d}"
            records.append(dict(request_id=rid, sample_id=article['start'], dataset_row_index=article['start'],
                dataset_row_end_exclusive=article['stop'], document_id=rid, article_title=article['title'],
                document_sha256=article['document_sha256'], prompt=article['document'],
                prompt_sha256=article['document_sha256'], prompt_token_count=3072,
                prompt_token_ids_sha256=token_sha(ids), original_document_token_count=len(article['ids'])))
            prompts.append(ids)
        workload = dict(schema=old['schema'], source_requests=records, actual_prompt_token_ids=prompts,
            arrival_traces_s=deepcopy(old['arrival_traces_s']), arrival_rule=old['arrival_rule'],
            document_identity_scope=provenance['identity_scope'])
        config = deepcopy(base)
        config.update(source=dict(provenance, cohort_id=label),
                      workload_sha256=sha(json.dumps(workload, sort_keys=True).encode()),
                      input_preparation='offline holdout prefix inputs; GPU UNRUN; inherited SLO values are '
                                        'compatibility fields, not newly selected business acceptance thresholds')
        folder = args.output_dir / label
        folder.mkdir()
        for name, data in [('config.json', config), ('workload.json', workload)]:
            path = folder / name
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            files[str(path.relative_to(args.output_dir))] = file_sha(path)
        cohorts[label] = dict(requests=32, workload_sha256=config['workload_sha256'],
            first_row=records[0]['dataset_row_index'], last_row_end_exclusive=records[-1]['dataset_row_end_exclusive'],
            articles=[{k: row[k] for k in ('document_id', 'dataset_row_index', 'dataset_row_end_exclusive',
                      'article_title', 'document_sha256', 'original_document_token_count')} for row in records])
    report = dict(status='INPUTS_PREPARED_GPU_UNRUN', cohorts=cohorts, provenance=provenance,
        files_sha256=files, script_sha256=file_sha(Path(__file__)), command=[sys.executable] + sys.argv,
        local_assets={'dataset_arrow': str(args.dataset_arrow.resolve()), 'model_files': local_model},
        old_input_files_sha256={name: file_sha(args.old_inputs / name) for name in ('config.json', 'workload.json')},
        duplicate_articles_skipped=duplicate_skips,
        checks={'old_32_documents_and_tokens_reproduced': True, 'old_and_new_document_hashes_unique_96': True,
                'old_and_new_prompt_hashes_unique_96': True, 'all_source_row_intervals_disjoint': True,
                'all_selected_articles_closed_by_next_title': True, 'same_prompt_output_lengths': True,
                'same_arrival_trace': True, 'no_network_or_gpu_required': True})
    (args.output_dir / 'inputs_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=report['status'], checks=report['checks'],
                         cohorts={label: {k: v for k, v in data.items() if k != 'articles'}
                                  for label, data in cohorts.items()})))


if __name__ == '__main__':
    main()
