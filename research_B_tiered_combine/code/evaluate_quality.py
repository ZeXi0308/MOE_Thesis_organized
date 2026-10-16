"""Article-level vLLM quality sweeps; no latency or native-EP claim.

Run under the shared GPU lock. The default arms compare Mode I with native
reduction and Mode II with grouped BF16. Each window is evaluated through all
arms before releasing its full-logit reference. All raw metrics are appended
after each completed arm; failures leave completed records intact.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import time
from quality_protocol import default_arms,payload_bytes,ppl,token_windows,validate_arms


def digest(data):return hashlib.sha256(data).hexdigest()


def append_record(handle,row):
    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    handle.flush();os.fsync(handle.fileno())


def checkpoint(path,row):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--articles',required=True)
    p.add_argument('--output',required=True);p.add_argument('--layers',type=int,required=True)
    p.add_argument('--top-k',type=int,required=True);p.add_argument('--hidden',type=int,required=True)
    p.add_argument('--context',type=int,default=512);p.add_argument('--ep-size',type=int,default=2)
    p.add_argument('--limit-docs',type=int,default=0)
    p.add_argument('--start-doc',type=int,default=0,help='continue at this source article index, preserving earlier run')
    p.add_argument('--max-article-tokens',type=int,default=0,help='0 means entire article; positive means explicit prefix limit')
    p.add_argument('--arms',help='JSON list in baseline-group order, optionally per-layer overrides')
    p.add_argument('--gpu-memory-utilization',type=float,default=.5)
    p.add_argument('--purpose',choices=['integration','calibration','heldout'],required=True)
    a=p.parse_args()
    if min(a.layers,a.top_k,a.hidden)>0 and a.hidden%16==0 and a.context>=2:pass
    else:raise ValueError('invalid dimensions/context')
    if a.limit_docs<0 or a.max_article_tokens<0 or a.start_doc<0:raise ValueError('negative limits')
    source=pathlib.Path(a.articles).read_bytes()
    docs=[json.loads(line) for line in source.decode().splitlines() if line.strip()]
    docs=docs[a.start_doc:]
    if a.limit_docs:docs=docs[:a.limit_docs]
    if not docs:raise ValueError('no articles')
    for doc in docs:
        if digest(doc['text'].encode())!=doc['text_sha256']:raise ValueError('article hash mismatch')
        if a.purpose=='heldout' and doc['split']!='test':raise ValueError('heldout requires test split')
        if a.purpose=='calibration' and doc['split']!='validation':raise ValueError('calibration requires validation split')
    arms=json.loads(pathlib.Path(a.arms).read_text()) if a.arms else default_arms(a.top_k,a.ep_size)
    validate_arms(arms)
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    model_config=json.loads((pathlib.Path(a.model)/'config.json').read_text())
    if model_config['hidden_size']!=a.hidden:raise ValueError('hidden size disagrees with checkpoint')
    config_k=model_config.get('num_experts_per_tok',model_config.get('num_experts_per_token'))
    if config_k is not None and config_k!=a.top_k:raise ValueError('top-k disagrees with checkpoint')
    manifest={'status':'RUNNING','arguments':vars(a),'model_config':model_config,
        'articles_sha256':digest(source),'selected_article_ids':[d['id'] for d in docs],
        'arms':arms,'versions':{name:importlib.metadata.version(name) for name in ['torch','triton','vllm']},
        'protocol':{'context':'one-token overlap; reset history at each block; no added BOS/EOS',
            'nll':'every token after article first, or after first of explicit prefix; teacher-forced block-context PPL',
            'kl':'full compute_logits distributions, includes final next-token prediction; different population from NLL',
            'native_ep':False,'mode_II':'all logical owner partials quantized on one GPU, including local partial',
            'byte_count':'logical payload and scale bytes; routing/control metadata excluded; no native wire claim',
            'timing':'quality instrumentation only'},'started_unix':time.time()}
    checkpoint(out/'manifest.json',manifest)
    from vllm import LLM,SamplingParams
    set_policy='tiered_set_policy';finish_policy='tiered_finish_policy';capture_control='tiered_capture_control'
    try:
        llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            moe_backend='triton',enable_prefix_caching=False,max_model_len=a.context+1,
            worker_extension_cls='quality_worker.QualityWorkerExtension',
            max_num_seqs=1,max_num_batched_tokens=a.context+1,
            gpu_memory_utilization=a.gpu_memory_utilization,seed=1701)
    except BaseException as exc:
        manifest.update(status='FAILED',error=repr(exc),finished_unix=time.time())
        checkpoint(out/'manifest.json',manifest)
        raise
    tokenizer=llm.get_tokenizer()
    sampling=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0)
    completed=[]
    try:
        with (out/'windows.jsonl').open('x') as windows_file,(out/'metrics.jsonl').open('x') as metrics_file:
            for doc in docs:
                tokens=list(tokenizer.encode(doc['text'],add_special_tokens=False))
                original_length=len(tokens)
                if a.max_article_tokens:tokens=tokens[:a.max_article_tokens]
                if len(tokens)<2:raise ValueError(f"{doc['id']}: fewer than 2 tokens")
                sums={arm['name']:{'nll_sum':0.,'tokens':0,'kl_sum':0.,'kl_positions':0,'windows':0} for arm in arms}
                for window_index,window in enumerate(token_windows(tokens,a.context)):
                    key={'article_id':doc['id'],'window':window_index}
                    append_record(windows_file,{**key,**window,'article_text_sha256':doc['text_sha256'],
                        'original_article_tokens':original_length,'evaluated_article_tokens':len(tokens)})
                    native_logps=None
                    for arm in arms:
                        name=arm['name'];baseline=arm['baseline']==name
                        installation=llm.collective_rpc(set_policy,args=(arm['policy'],a.layers,
                            arm.get('layer_policies'),arm.get('profile',False)))
                        llm.collective_rpc(capture_control,args=('baseline' if baseline else 'candidate',))
                        results=llm.generate([{'prompt_token_ids':window['token_ids']}],sampling,use_tqdm=False)
                        if len(results)!=1:raise RuntimeError('unexpected request count')
                        result=results[0]
                        if list(result.prompt_token_ids)!=window['token_ids']:raise RuntimeError('prompt token mismatch')
                        if len(result.prompt_logprobs)!=len(window['token_ids']):raise RuntimeError('prompt logprob length mismatch')
                        logps=[]
                        for pos,(token,scores) in enumerate(zip(result.prompt_token_ids,result.prompt_logprobs)):
                            if pos==0:
                                if scores is not None:raise RuntimeError('unexpected first-token logprob')
                            elif scores is None:raise RuntimeError('missing true-token logprob')
                            else:logps.append(float(scores[token].logprob))
                        captures=llm.collective_rpc(capture_control,args=('finish',))
                        if len(captures)!=1:raise RuntimeError('quality evaluation requires one worker')
                        capture=captures[0]
                        diagnostics=llm.collective_rpc(finish_policy,args=(a.layers,))
                        if name=='native':native_logps=logps
                        if name=='I_bf16':
                            if native_logps is None or len(logps)!=len(native_logps):raise RuntimeError('native identity baseline missing')
                            if max(abs(x-y) for x,y in zip(logps,native_logps))>=1e-5:raise RuntimeError('identity NLL mismatch')
                            if not capture['token_kl'] or max(abs(x) for x in capture['token_kl'])>=1e-6:raise RuntimeError('identity KL mismatch')
                        row={**key,'arm':name,'baseline':arm['baseline'],'policy':arm['policy'],
                            'layer_policies':arm.get('layer_policies'),
                            'true_token_logprobs':logps,'nll_sum':-sum(logps),'tokens':len(logps),
                            'distribution_kl':capture,'worker':diagnostics,'installation':installation,
                            'output_token_ids':list(result.outputs[0].token_ids),
                            'finish_reason':result.outputs[0].finish_reason,
                            'logical_payload_bytes_per_layer':None if arm.get('layer_policies') else
                                payload_bytes(arm['policy'],len(window['token_ids']),a.hidden,a.top_k)}
                        append_record(metrics_file,row)
                        s=sums[name];s['nll_sum']+=row['nll_sum'];s['tokens']+=len(logps)
                        s['kl_sum']+=sum(capture['token_kl']);s['kl_positions']+=len(capture['token_kl']);s['windows']+=1
                    print(json.dumps({**key,'status':'WINDOW_COMPLETE','arms':len(arms)}),flush=True)
                for name,s in sums.items():
                    if s['tokens']!=len(tokens)-1:raise RuntimeError(f'{name}: NLL token accounting mismatch')
                    s['ppl']=ppl(s['nll_sum'],s['tokens'])
                    s['mean_kl']=s['kl_sum']/s['kl_positions'] if s['kl_positions'] else None
                completed.append({'article_id':doc['id'],'original_tokens':original_length,
                    'evaluated_tokens':len(tokens),'arms':sums})
                checkpoint(out/'documents.json',completed)
    except BaseException as exc:
        manifest.update(status='FAILED',error=repr(exc),finished_unix=time.time())
        checkpoint(out/'manifest.json',manifest)
        raise
    finally:
        cleanup_errors=[]
        for func,args in [(set_policy,(None,a.layers)),(capture_control,('restore',))]:
            try:llm.collective_rpc(func,args=args)
            except Exception as exc:cleanup_errors.append(repr(exc))
        if cleanup_errors:
            manifest.update(status='FAILED',cleanup_errors=cleanup_errors,finished_unix=time.time())
            checkpoint(out/'manifest.json',manifest)
    if cleanup_errors:raise RuntimeError(f'Worker cleanup failed: {cleanup_errors}')
    manifest.update(status='COMPLETE',completed_documents=len(completed),finished_unix=time.time())
    checkpoint(out/'manifest.json',manifest)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
