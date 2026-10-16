#!/usr/bin/env python3
"""Development document families with Instruct chat and natural EOS.

Uses the existing B source documents, not repeated random tokens. The fixed
prompt-length mixture is a diagnostic workload; it is not a deployment trace.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-model',type=Path,help='Tokenizer that produced source token IDs; defaults to model')
    p.add_argument('--disable-thinking',action='store_true',help='Qwen non-thinking template and recommended sampling')
    p.add_argument('--family',choices=('W1','W2','W3'),default='W2')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=int,default=256)
    args=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
    source_tokenizer=(AutoTokenizer.from_pretrained(str(args.source_model),local_files_only=True)
                      if args.source_model else tokenizer)
    old=json.loads(args.source.read_text());rows=[];prompts=[]
    pairs=list(zip(old['source_requests'],old['actual_prompt_token_ids']))[:args.requests]
    if args.family=='W3':
        # Preserve W2's documents and lengths, change their temporal grouping.
        pairs.sort(key=lambda pair:-len(pair[1]))
    for i,(row,ids) in enumerate(pairs):
        if i>=args.requests:break
        length=(512 if args.family=='W1' else min(3072,len(ids)) if args.family=='W3'
                else 3072 if i%2==0 else 512)
        # Preserve the generation suffix and all instruction tokens while
        # truncating only the source article to the desired total bound.
        count=min(len(ids),length)
        while True:
            document=source_tokenizer.decode(ids[:count],skip_special_tokens=True)
            content=('Read the following document. Give a detailed, structured analysis of its main claims, '
                     'supporting facts, implications, and uncertainties. Use only information supported by the document.\n\n'+document)
            tokens=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,return_dict=False,
                **({'enable_thinking':False} if args.disable_thinking else {}))
            if len(tokens)<=length:break
            count-=max(1,len(tokens)-length)
            if count<=0:raise RuntimeError('Instruction exceeds prompt limit')
        rid=f'{args.family.lower()}-{i:04d}'
        rows.append(dict(request_id=rid,document_id=row['document_id'],source_request_id=row['request_id'],
            prompt_token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()))
        prompts.append(tokens)
    if len(rows)!=args.requests:raise ValueError('Insufficient source requests')
    payload=dict(config=dict(model_path=str(args.model),model_revision=args.model.name,workload=args.family,
        source_tokenizer_path=str(args.source_model or args.model),
        chat_template_kwargs={'enable_thinking':False} if args.disable_thinking else {},
        sampling={'temperature':.7,'top_p':.8,'top_k':20,'seed':20261016} if args.disable_thinking else {'temperature':0.0},
        family_definition={'W1':'short documents up to 512 tokens','W2':'alternating 3072/512-token documents',
                           'W3':'W2 documents and lengths, long-prefix requests first then short'}[args.family],
        source_path=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        requests=len(rows),seed=20261016,output_tokens=1024,ignore_eos=False,min_tokens=0),
        workload=dict(source_requests=rows,actual_prompt_token_ids=prompts,
                      arrival_traces_s={'steady':[i*.1 for i in range(len(rows))]}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(payload,f)
    print(json.dumps(dict(requests=len(rows),prompt_min=min(map(len,prompts)),prompt_max=max(map(len,prompts)),
                         output=str(args.output))))


if __name__=='__main__':main()
