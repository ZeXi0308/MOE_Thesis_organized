"""Full OLMoE native/identity KL and decode policy coverage smoke."""
import argparse,json,pathlib,time


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,
        enforce_eager=True,compilation_config={'mode':0,'cudagraph_mode':'NONE'},
        worker_cls='skip_worker.ExpertSkipWorker',moe_backend='triton',
        enable_prefix_caching=False,max_model_len=512,max_num_seqs=8,
        max_num_batched_tokens=512,gpu_memory_utilization=.35,seed=1701)
    prompts=['The purpose of a scientific experiment is to distinguish competing explanations using observations.',
             'Explain how a mixture of experts combines the outputs of its selected expert networks.']
    rows=[]
    arms=[('native',None),('dense',{'mode':'dense'}),('global4',{'mode':'global','drop_j':4,'mass_budget':1.}),
          ('gate',{'mode':'gate','gate_threshold':.15,'mass_budget':.25}),
          ('marginal',{'mode':'marginal','tail_rank':4,'gate_threshold':.15,'mass_budget':.25})]
    for name,spec in arms[:2]:
        installed=llm.collective_rpc('skip_set_policy',args=(spec,'all',0,False))
        llm.collective_rpc('skip_capture_control',args=('baseline' if name=='native' else 'candidate',))
        results=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0),use_tqdm=False)
        kl=llm.collective_rpc('skip_capture_control',args=('finish',))
        row={'arm':name,'scope':'all','installation':installed,'kl':kl,
             'worker':llm.collective_rpc('skip_snapshot'),
             'outputs':[{'tokens':list(r.outputs[0].token_ids),'text':r.outputs[0].text} for r in results]}
        rows.append(row);out.write_text(json.dumps({'status':'RUNNING','rows':rows},indent=2))
        if name=='dense':
            assert row['outputs']==rows[0]['outputs']
            assert all(max(abs(x) for x in c['token_kl'])<1e-6 for c in kl)
    llm.collective_rpc('skip_capture_control',args=('restore',))
    for B in [1,8]:
        batch=[prompts[i%2]+f' Give example number {i+1}.' for i in range(B)]
        for name,spec in arms[1:]:
            installed=llm.collective_rpc('skip_set_policy',args=(spec,'decode',128,True))
            results=llm.generate(batch,SamplingParams(temperature=0,max_tokens=8,ignore_eos=True),use_tqdm=False)
            stats=llm.collective_rpc('skip_snapshot')
            row={'arm':name,'batch':B,'scope':'decode','installation':installed,'worker':stats,
                 'outputs':[{'tokens':list(r.outputs[0].token_ids),'text':r.outputs[0].text,'finish_reason':r.outputs[0].finish_reason} for r in results]}
            rows.append(row);out.write_text(json.dumps({'status':'RUNNING','rows':rows},indent=2))
            for s in stats:
                assert s['phase_steps'].get('decode',0)>0
                assert len(s['layers']['layers'])==16
                assert all(r['forward_completed'] for r in s['trace'])
    out.write_text(json.dumps({'status':'PASS','scope':'integration only; eager tracing changes timing; short forced-length generation is not task quality',
                               'rows':rows,'completed_unix':time.time()},indent=2))
    print('PASS',flush=True)


if __name__=='__main__':main()
