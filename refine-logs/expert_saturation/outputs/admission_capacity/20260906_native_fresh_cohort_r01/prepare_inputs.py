#!/usr/bin/env python3
"""Prepare one next-row cohort offline, without model loading or outcome access."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUTPUTS = ROOT / 'refine-logs/expert_saturation/outputs/admission_capacity'


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token_digest(ids):
    return digest(json.dumps(ids, separators=(',', ':')).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-arrow', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    manifest = read(ROOT / 'docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json')
    base_dir = OUTPUTS / '20260905_pre_gpu_r01/prepared'
    base, base_workload = read(base_dir / 'config.json'), read(base_dir / 'workload.json')
    prior_paths = [base_dir / 'workload.json'] + [
        OUTPUTS / f'20260906_cohort_probe_r01/offset{offset}_inputs/workload.json'
        for offset in (16, 32)]
    prior = [read(path) for path in prior_paths]
    prior_text = {r['prompt_sha256'] for w in prior for r in w['source_requests']}
    prior_tokens = {token_digest(ids) for w in prior for ids in w['actual_prompt_token_ids']}
    if len(prior_text) != 48 or len(prior_tokens) != 48:
        raise ValueError('the frozen three prior cohorts do not contain 48 distinct inputs')
    arrow_sha = digest(args.dataset_arrow.read_bytes())
    if arrow_sha != manifest['dataset']['arrow_sha256']:
        raise ValueError('cached Arrow bytes differ from the existing dataset')
    dataset = Dataset.from_file(str(args.dataset_arrow))
    model = base['model']
    tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=model['tokenizer_revision'],
                                             local_files_only=True)
    for name, expected in manifest['tokenizer']['files'].items():
        path = Path(cached_file(model['id'], name, revision=model['tokenizer_revision'], local_files_only=True))
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'pinned tokenizer file changed: {name}')
    if tokenizer.truncation_side != 'right':
        raise ValueError('tokenizer truncation side changed')
    start = 1 + max(r['dataset_row_index'] for r in manifest['requests'])
    rows, tokens, selected_text, selected_tokens = [], [], set(), set()
    for index in range(start, len(dataset)):
        prompt = str(dataset[index]['text'])
        if not prompt.strip():
            continue
        ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                        max_length=base['prompt_tokens'], padding=False)['input_ids']
        if len(ids) != base['prompt_tokens']:
            continue
        text_sha, ids_sha = digest(prompt.encode()), token_digest(ids)
        if text_sha in prior_text | selected_text or ids_sha in prior_tokens | selected_tokens:
            continue
        rows.append(dict(request_id=f'capacity-fresh-row-{index:06d}', sample_id=index,
            dataset_row_index=index, document_id=f'wikitext-test-row-{index:06d}-{text_sha[:16]}',
            document_sha256=text_sha, prompt=prompt, prompt_sha256=text_sha,
            prompt_token_count=len(ids), prompt_token_ids_sha256=ids_sha))
        tokens.append(ids)
        selected_text.add(text_sha)
        selected_tokens.add(ids_sha)
        if len(rows) == base['requests']:
            break
    if len(rows) != 16:
        raise ValueError('not enough new eligible dataset rows')
    workload = dict(schema='olmoe-admission-inputs-v1', source_requests=rows,
        actual_prompt_token_ids=tokens, arrival_traces_s=base_workload['arrival_traces_s'],
        arrival_rule=base_workload['arrival_rule'],
        document_identity_scope='new rows/text/token IDs within admission_capacity only; not certified article-disjoint')
    config = {k: base[k] for k in ('model', 'requests', 'prompt_tokens', 'output_tokens',
                                  'seed', 'ttft_slo_s', 'tpot_slo_s', 'arrival_gap_s', 'burst_size')}
    config.update(workload_sha256=digest(json.dumps(workload, sort_keys=True).encode()),
        input_preparation='native-only inputs; campaign declares execution plans and primary SLO',
        source=dict(dataset={k: manifest['dataset'][k] for k in ('id', 'config', 'split', 'revision')},
            arrow_sha256=arrow_sha, observed_dataset_fingerprint=str(dataset._fingerprint),
            selection_rule='first 16 eligible nonduplicate rows after the old manifest last row',
            start_dataset_row_index=start, selected_dataset_rows=[r['dataset_row_index'] for r in rows],
            excluded_prior_request_count=48, prior_workload_files=[str(p.relative_to(ROOT)) for p in prior_paths],
            tokenizer_files_sha256=manifest['tokenizer']['files'],
            freshness_scope=workload['document_identity_scope']))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (('config.json', config), ('workload.json', workload)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='INPUTS_PREPARED_GPU_UNRUN', selected_rows=config['source']['selected_dataset_rows'],
        workload_sha256=config['workload_sha256'], actual_fingerprint=str(dataset._fingerprint))))


if __name__ == '__main__':
    main()
