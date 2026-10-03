#!/usr/bin/env python3
"""Offline paired natural-document inputs; no padding, repetition or cross-article joining."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
REVISION = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-arrow', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--long-prompt-tokens', type=int, default=3072)
    parser.add_argument('--output-tokens', type=int, default=1024)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    base = json.loads((ROOT / 'refine-logs/expert_saturation/outputs/admission_capacity/'
                       '20260905_pre_gpu_r01/prepared/config.json').read_text())
    model = base['model']
    assert model['revision'] == model['tokenizer_revision'] == REVISION
    model_config_path = Path(cached_file(model['id'], 'config.json', revision=REVISION, local_files_only=True))
    model_config = json.loads(model_config_path.read_text())
    assert 128 < args.long_prompt_tokens
    assert 0 < args.output_tokens <= model_config['max_position_embeddings'] - args.long_prompt_tokens
    tokenizer = AutoTokenizer.from_pretrained(model['id'], revision=REVISION, local_files_only=True)
    manifest = json.loads((ROOT / 'docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json').read_text())
    for name, expected in manifest['tokenizer']['files'].items():
        assert file_sha(Path(cached_file(model['id'], name, revision=REVISION, local_files_only=True))) == expected
    assert args.dataset_arrow.name == 'wikitext-train-00000-of-00002.arrow'
    assert args.dataset_arrow.parent.name == DATASET_REVISION
    assert args.dataset_arrow.parents[2].name == 'wikitext-103-raw-v1'
    dataset = Dataset.from_file(str(args.dataset_arrow))
    selected, start, pieces = [], None, []
    for index, row in enumerate(dataset):
        text = str(row['text'])
        if re.fullmatch(r'= [^=].*? =', text.strip()):
            if start is not None:
                document = ''.join(pieces)
                ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)['input_ids']
                if len(ids) >= args.long_prompt_tokens:
                    selected.append(dict(start=start, stop=index, document=document, ids=ids,
                                         title=pieces[0].strip(), document_sha256=sha(document.encode())))
                    if len(selected) == 32:
                        break
            start, pieces = index, []
        if start is not None:
            pieces.append(text)
    # A shard's unterminated last article is intentionally excluded.
    assert len(selected) == 32, 'fewer than 32 complete eligible articles in this shard'
    assert len({x['document_sha256'] for x in selected}) == 32
    assert len({token_sha(x['ids'][:128]) for x in selected}) == 32
    provenance = dict(dataset_id='wikitext', dataset_config='wikitext-103-raw-v1', split='train',
        dataset_revision=DATASET_REVISION, arrow_file=args.dataset_arrow.name, arrow_sha256=file_sha(args.dataset_arrow),
        observed_dataset_fingerprint=str(dataset._fingerprint), model_config_sha256=file_sha(model_config_path),
        model_max_position_embeddings=model_config['max_position_embeddings'], tokenizer_files_sha256=manifest['tokenizer']['files'],
        selection_rule=f'first 32 complete articles with >={args.long_prompt_tokens} tokens in train shard 0, in source row order',
        pre_run_length_rationale='3072 prompt +1024 output retains active requests longer while later long prefills are admitted; 3968+128 could drain before 32 requests become active. This is regime qualification, chosen before GPU results.',
        reconstruction_rule='concatenate original row strings exactly from one top-level title to before the next; no added separators',
        identity_scope='article identities within this probe; operating-regime qualification, not a fresh holdout')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for label, count in [('short', 128), ('long', args.long_prompt_tokens)]:
        records, prompts = [], []
        for article in selected:
            ids = article['ids'][:count]
            rid = f"memory-train-article-{article['start']:07d}"
            records.append(dict(request_id=rid, sample_id=article['start'], dataset_row_index=article['start'],
                dataset_row_end_exclusive=article['stop'], document_id=rid, article_title=article['title'],
                document_sha256=article['document_sha256'], prompt=article['document'],
                prompt_sha256=article['document_sha256'], prompt_token_count=count, prompt_token_ids_sha256=token_sha(ids),
                original_document_token_count=len(article['ids'])))
            prompts.append(ids)
        workload = dict(schema='olmoe-admission-inputs-v1', source_requests=records, actual_prompt_token_ids=prompts,
            arrival_traces_s={'steady': [round(i * .05, 8) for i in range(32)]}, arrival_rule='steady: request i arrives at i*0.05 seconds',
            document_identity_scope=provenance['identity_scope'])
        config = dict(model=model, requests=32, prompt_tokens=count, output_tokens=args.output_tokens, seed=base['seed'],
            ttft_slo_s=5.0, tpot_slo_s=.2, arrival_gap_s=.05, burst_size=8, source=provenance,
            workload_sha256=sha(json.dumps(workload, sort_keys=True).encode()),
            input_preparation='native-only paired prefix inputs; GPU unrun; campaign declares SLO and engine settings')
        folder = args.output_dir / label
        folder.mkdir()
        for name, data in [('config.json', config), ('workload.json', workload)]:
            (folder / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        outputs[label] = workload
    assert all(a == b[:128] for a, b in zip(outputs['short']['actual_prompt_token_ids'], outputs['long']['actual_prompt_token_ids']))
    report = dict(status='INPUTS_PREPARED_GPU_UNRUN', requests=32, short_prompt_tokens=128, long_prompt_tokens=args.long_prompt_tokens,
        output_tokens=args.output_tokens, arrival_horizon_s=1.55, provenance=provenance,
        selected_articles=[{k: x[k] for k in ('start', 'stop', 'title', 'document_sha256')} for x in selected],
        checks={'unique_articles': True, 'unique_short_prefixes': True, 'short_is_long_prefix': True,
                'context_legal': True, 'same_arrivals': True, 'whole_article_source_boundaries': True})
    (args.output_dir / 'inputs_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('selected_articles', 'provenance')}))


if __name__ == '__main__':
    main()
