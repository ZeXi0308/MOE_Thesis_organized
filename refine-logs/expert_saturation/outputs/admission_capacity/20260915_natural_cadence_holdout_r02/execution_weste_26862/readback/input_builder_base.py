#!/usr/bin/env python3
"""CPU-only source-ordered full-article candidate; arrivals remain unfrozen."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

TITLE = re.compile(r' = [^=\n]+ = \n')
REVISION = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def token_sha(ids):
    return digest(json.dumps(ids, separators=(',', ':')).encode())


def describe(values):
    values = sorted(values)
    def quantile(q):
        x = (len(values)-1)*q; lo = math.floor(x); hi = math.ceil(x)
        return values[lo] + (values[hi]-values[lo])*(x-lo)
    return dict(count=len(values), total=sum(values), minimum=min(values), maximum=max(values),
        mean=statistics.mean(values), median=statistics.median(values),
        p10=quantile(.1), p25=quantile(.25), p75=quantile(.75), p90=quantile(.9),
        population_std=statistics.pstdev(values))


def closed_articles(dataset):
    start, pieces, previous = None, [], ''
    for i, row in enumerate(dataset):
        text = str(row['text'])
        title = (bool(TITLE.fullmatch(text)) and not previous.strip() and i+1<len(dataset)
                 and not str(dataset[i+1]['text']).strip())
        if title:
            if start is not None:
                yield start, i, ''.join(pieces)
            start, pieces = i, []
        if start is not None: pieces.append(text)
        previous = text


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--dataset-arrow',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists(): raise SystemExit('preserve candidate; output directory already exists')
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    prior_root=args.source_root/'refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_inputs_r01'
    prior_dirs=[prior_root/'prior_inputs'/n for n in ('original','cohort0','cohort1','cohort2')]+[prior_root/'prepared']
    used=[]; prior_receipts=[]
    for folder in prior_dirs:
        cfg=json.loads((folder/'config.json').read_text());work=json.loads((folder/'workload.json').read_text())
        assert digest(json.dumps(work,sort_keys=True).encode())==cfg['workload_sha256']
        assert len(work['source_requests'])==32
        used.extend(work['source_requests'])
        prior_receipts.append(dict(path=str(folder),workload_sha256=file_sha(folder/'workload.json')))
    used_ids={r['document_id'] for r in used};used_hashes={r['document_sha256'] for r in used}
    assert len(used_ids)==len(used_hashes)==160
    natural=args.source_root/'refine-logs/independent_ideas_20260911/natural_prefill_tails_r01/prepared/natural'
    cached_work=json.loads((natural/'workload.json').read_text());cfg=json.loads((natural/'config.json').read_text())
    assert digest(json.dumps(cached_work,sort_keys=True).encode())==cfg['workload_sha256']
    model, provenance=cfg['model'],cfg['source']
    assert model['revision']==model['tokenizer_revision']==REVISION
    assert provenance['dataset_revision']==DATASET_REVISION
    assert args.dataset_arrow.name==provenance['arrow_file']
    assert file_sha(args.dataset_arrow)==provenance['arrow_sha256']
    local_assets={}
    for name,expected in dict(provenance['tokenizer_files_sha256'],**{'config.json':provenance['model_config_sha256']}).items():
        path=Path(cached_file(model['id'],name,revision=REVISION,local_files_only=True))
        assert file_sha(path)==expected
        local_assets[name]=dict(path=str(path),sha256=expected)
    tokenizer=AutoTokenizer.from_pretrained(model['id'],revision=REVISION,local_files_only=True)
    cached={r['document_id']:(r,ids) for r,ids in zip(cached_work['source_requests'],cached_work['actual_prompt_token_ids'])}
    dataset=Dataset.from_file(str(args.dataset_arrow));records=[];prompts=[];scan=[];reused=[]
    selected_hashes=set();selected_tokens=set();skips=Counter()
    for start,end,text in closed_articles(dataset):
        rid=f'memory-train-article-{start:07d}';doc_hash=digest(text.encode())
        if rid in cached:
            old,ids=cached[rid]
            assert text==old['prompt'] and end==old['dataset_row_end_exclusive']
            assert doc_hash==old['document_sha256'] and token_sha(ids)==old['prompt_token_ids_sha256']
            assert len(ids)==old['original_document_token_count']
            reused.append(rid)
        else:
            ids=tokenizer(text,add_special_tokens=True,truncation=False,padding=False)['input_ids']
        overlap=any(max(start,r['dataset_row_index'])<min(end,r['dataset_row_end_exclusive']) for r in used)
        reason=('prior_identity_or_row' if rid in used_ids or doc_hash in used_hashes or overlap else
                'outside_fixed_256_3072' if not 256<=len(ids)<=3072 else
                'duplicate' if doc_hash in selected_hashes or token_sha(ids) in selected_tokens else 'selected')
        scan.append(dict(document_id=rid,row_start=start,row_end_exclusive=end,
            original_document_token_count=len(ids),reason=reason))
        skips[reason]+=1
        if reason!='selected': continue
        selected_hashes.add(doc_hash);selected_tokens.add(token_sha(ids))
        records.append(dict(request_id=rid,document_id=rid,dataset_row_index=start,
            dataset_row_end_exclusive=end,article_title=text.splitlines()[0].strip(),
            source_index=len(records),prompt=text,document_sha256=doc_hash,prompt_sha256=doc_hash,
            prompt_token_count=len(ids),original_document_token_count=len(ids),
            prompt_token_ids_sha256=token_sha(ids),max_output_tokens=1024))
        prompts.append(ids)
        if len(records)==64: break
    assert len(records)==64 and len(selected_hashes)==len(selected_tokens)==64
    lengths=[len(ids) for ids in prompts];reserved=[math.ceil((n+1024)/16) for n in lengths]
    windows=[dict(first_source_index=i,last_source_index=i+31,reserved_blocks=sum(reserved[i:i+32])) for i in range(33)]
    workload=dict(schema='olmoe-natural-streaming-candidate-v1',source_requests=records,
        actual_prompt_token_ids=prompts,arrival_traces_s=None,arrival_rule='UNFROZEN',
        output_contract=dict(max_output_tokens=1024,ignore_eos=False,min_tokens=0,
                             actual_output_lengths='UNKNOWN_UNTIL_EXECUTION'))
    config=dict(status='CPU_CANDIDATE_ARRIVAL_UNFROZEN_GPU_UNRUN',model=model,requests=64,
        prompt_tokens=None,prompt_tokens_by_request=lengths,output_tokens=None,max_output_tokens=1024,
        arrival_gap_s=None,arrival_process=None,max_model_len=4096,
        workload_sha256=digest(json.dumps(workload,sort_keys=True).encode()),
        source=dict(dataset_id='wikitext',dataset_config='wikitext-103-raw-v1',split='train',
            dataset_revision=DATASET_REVISION,arrow_file=args.dataset_arrow.name,
            arrow_sha256=provenance['arrow_sha256'],tokenizer_files_sha256=provenance['tokenizer_files_sha256'],
            selection='First64 source-ordered complete articles after excluding160 document identities/hashes/overlapping rows; full token length256..3072. No latency/GPU/actual EOS or output length selection.',
            reconstruction_rule=provenance['reconstruction_rule']))
    report=dict(status=config['status'],command=[sys.executable]+sys.argv,prior_inputs=prior_receipts,
        excluded_prior_documents=160,excluded_prior_max_row_end=21131,local_assets=local_assets,
        cached_full_documents_available=16,reused_full_documents=reused,
        source_interval=dict(first_row=scan[0]['row_start'],end_exclusive=scan[-1]['row_end_exclusive']),
        examined_article_counts=dict(skips),examined_articles=scan,
        natural_lengths_all_closed_articles_in_examined_interval=describe([r['original_document_token_count'] for r in scan]),
        natural_lengths_unused_articles_in_examined_interval=describe([r['original_document_token_count'] for r in scan if r['reason']!='prior_identity_or_row']),
        selected_full_prompt_lengths=describe(lengths),selected_lengths_source_order=lengths,
        conservative_block_upper_bounds=dict(formula='ceil((P+1024)/16)',physical_budget_reference=6656,
            contiguous_32_windows=windows,contiguous_32_min=min(w['reserved_blocks'] for w in windows),
            contiguous_32_max=max(w['reserved_blocks'] for w in windows),
            any_subset_up_to_32_max=sum(sorted(reserved,reverse=True)[:32]),
            boundary='Declared max-output capacity upper bounds, not actual EOS requirements. Contiguous windows do not cover all resident sets under heterogeneous completion; largest32 sum does.',
            no_gpu_or_latency_selection=True),
        runtime_gap='Existing native_capture still forces min_tokens=max_tokens and ignore_eos=True with length-only completion assertions; candidate requires an EOS-capable capture adapter before GPU use.',
        unchosen=['arrival process','arrival scale','episode launch','hardware allocation','business SLO'],
        gpu_executions=0,network_downloads=0)
    args.output_dir.mkdir(parents=True)
    for name,obj in [('workload.json',workload),('config.json',config),('INPUT_STATS.json',report)]:
        (args.output_dir/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=config['status'],prompt_lengths=report['selected_full_prompt_lengths'],
        source_interval=report['source_interval'],counts=report['examined_article_counts'],
        contiguous32_min=report['conservative_block_upper_bounds']['contiguous_32_min'],
        contiguous32_max=report['conservative_block_upper_bounds']['contiguous_32_max'],
        any32_max=report['conservative_block_upper_bounds']['any_subset_up_to_32_max'])))


if __name__=='__main__':main()
