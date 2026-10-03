#!/usr/bin/env python3
"""Offline H inputs: first128 unused full train articles, never outcome-selected."""
import argparse
from collections import Counter
import json
from pathlib import Path
from input_builder_base import closed_articles, describe, digest, file_sha, token_sha, REVISION, DATASET_REVISION


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--dataset-arrow',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Preserve prepared inputs: output already exists')
    here=Path(__file__).resolve().parent
    inventory=json.loads((here/'used_documents.json').read_text())
    for receipt in inventory['source_receipts']:
        assert file_sha(args.source_root/receipt['path'])==receipt['sha256']
    used=inventory['train_articles']
    used_ids={r['document_id'] for r in used}
    used_hashes=set(inventory['global_document_hashes'])
    used_tokens=set(inventory['global_prompt_token_hashes'])
    assert len(used)==len(used_ids)==224
    base=args.source_root/'refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_recovery_cadence_r01/pkg/inputs/config.json'
    config=json.loads(base.read_text());source=config['source'];model=config['model']
    assert model['revision']==model['tokenizer_revision']==REVISION
    assert source['dataset_revision']==DATASET_REVISION
    assert args.dataset_arrow.name==source['arrow_file']
    assert file_sha(args.dataset_arrow)==source['arrow_sha256']
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    assets={}
    for name,expected in dict(source['tokenizer_files_sha256'],**{'config.json':'3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335'}).items():
        path=Path(cached_file(model['id'],name,revision=REVISION,local_files_only=True))
        assert file_sha(path)==expected
        assets[name]=dict(path=str(path),sha256=expected)
    tokenizer=AutoTokenizer.from_pretrained(model['id'],revision=REVISION,local_files_only=True)
    dataset=Dataset.from_file(str(args.dataset_arrow))
    rows=[];prompts=[];scan=[];counts=Counter();chosen_hashes=set();chosen_tokens=set()
    for start,end,text in closed_articles(dataset):
        rid=f'memory-train-article-{start:07d}';text_sha=digest(text.encode())
        overlap=any(max(start,r['dataset_row_index'])<min(end,r['dataset_row_end_exclusive']) for r in used)
        if rid in used_ids or text_sha in used_hashes or overlap:
            counts['prior_identity_text_or_rows']+=1
            scan.append(dict(document_id=rid,row_start=start,row_end_exclusive=end,reason='prior_identity_text_or_rows'))
            continue
        ids=tokenizer(text,add_special_tokens=True,truncation=False,padding=False)['input_ids']
        tok_sha=token_sha(ids)
        reason=('outside_fixed_256_3072' if not 256<=len(ids)<=3072 else
            'prior_token_content' if tok_sha in used_tokens else
            'duplicate_new_content' if text_sha in chosen_hashes or tok_sha in chosen_tokens else 'selected')
        counts[reason]+=1
        scan.append(dict(document_id=rid,row_start=start,row_end_exclusive=end,
            original_document_token_count=len(ids),reason=reason))
        if reason!='selected':
            continue
        chosen_hashes.add(text_sha);chosen_tokens.add(tok_sha)
        rows.append(dict(request_id=rid,document_id=rid,dataset_row_index=start,
            dataset_row_end_exclusive=end,article_title=text.splitlines()[0].strip(),
            source_index=len(rows),prompt=text,document_sha256=text_sha,prompt_sha256=text_sha,
            prompt_token_count=len(ids),original_document_token_count=len(ids),
            prompt_token_ids_sha256=tok_sha,max_output_tokens=1024))
        prompts.append(ids)
        if len(rows)==128:
            break
    assert len(rows)==len(chosen_hashes)==len(chosen_tokens)==128
    lengths=[len(ids) for ids in prompts]
    workload=dict(schema='olmoe-natural-cadence-holdout-v1',source_requests=rows,
        actual_prompt_token_ids=prompts,arrival_traces_s={'steady':[i*.2 for i in range(128)]},
        arrival_rule='Source order,0.2s spacing,128requests,25.4s finite arrival span',
        output_contract=dict(max_output_tokens=1024,ignore_eos=False,min_tokens=0,
            actual_output_lengths='UNKNOWN_UNTIL_EXECUTION'))
    for key in ('candidate_source','inherited_freeze_contract','freeze_contract'):
        config.pop(key,None)
    config.update(status='FROZEN_HOLDOUT_INPUT_GPU_UNRUN',requests=128,
        prompt_tokens=max(lengths),prompt_tokens_by_request=lengths,arrival_span_s=127*.2,
        workload_sha256=digest(json.dumps(workload,sort_keys=True).encode()),
        prompt_tokens_semantics=f'Upper bound{max(lengths)}; full lengths{min(lengths)}..{max(lengths)}, no padding/truncation.',
        input_preparation='First128 source-ordered unused complete train articles; exclude frozen224 prior train articles and known token contents. No future EOS/actions/latency selection.',
        input_independence='G is selection data; H is new document input and a longer finite arrival episode, not an independent model/runtime or host-turnover measurement.',
        exclusion_inventory_sha256=file_sha(here/'used_documents.json'))
    config['source']=dict(source,selection='First128 source-ordered complete articles after excluding224 prior train document identities/text hashes/overlapping rows and known token contents; full token length256..3072. No GPU, latency, actual EOS or action selection.')
    report=dict(status='CPU_PREPARED_GPU_UNRUN',builder_base_sha256=file_sha(here/'input_builder_base.py'),
        source_base_config_sha256=file_sha(base),exclusion_inventory_sha256=file_sha(here/'used_documents.json'),
        excluded_train_articles=len(used),excluded_prior_max_row_end=max(r['dataset_row_end_exclusive'] for r in used),
        local_assets=assets,source_arrow_sha256=source['arrow_sha256'],
        scanned_interval=dict(first_row=scan[0]['row_start'],end_exclusive=scan[-1]['row_end_exclusive']),
        examined_article_counts=dict(counts),examined_articles=scan,
        selected_full_prompt_lengths=describe(lengths),selected_row_first=rows[0]['dataset_row_index'],
        selected_row_end_exclusive=rows[-1]['dataset_row_end_exclusive'],
        arrival_gap_s=.2,arrival_span_s=127*.2,measured_requests_per_cell=128,
        measured_request_budget=768,gpu_executions=0,network_downloads=0)
    args.output_dir.mkdir(parents=True)
    for name,obj in [('workload.json',workload),('config.json',config),('INPUT_STATS.json',report)]:
        (args.output_dir/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('local_assets','examined_articles')}))


if __name__=='__main__':
    main()
