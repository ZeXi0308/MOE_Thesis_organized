"""Two-prompt integration smoke, NOT a corpus quality evaluation or latency test."""
import argparse,json,math,pathlib


def set_policy(worker, spec):
    import vllm_quality_hook as module
    from quality_transform import Policy
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    old=getattr(module,'_ACTIVE',None)
    if old is not None:
        old.__exit__(None,None,None);module._ACTIVE=None
    if spec is None:return {'installed':False}
    hook=module.QualityHook(Policy(**spec))
    backends={}
    for name,layer in worker.get_model().named_modules():
        method=getattr(layer,'quant_method',None)
        kernel=getattr(method,'moe_kernel',None)
        if kernel is None:continue
        expert=getattr(kernel,'fused_experts',None)
        if expert is None:continue
        backends[name]=type(expert).__name__
        if type(expert) is not TritonExperts:raise RuntimeError(f'Unsupported backend {name}: {type(expert)}')
        hook.bind(expert,name)
    if len(backends)!=16:raise RuntimeError(f'Expected 16 OLMoE layers: {backends}')
    hook.__enter__();module._ACTIVE=hook
    return {'installed':True,'backends':backends}


def coverage(worker):
    import vllm_quality_hook as module
    hook=getattr(module,'_ACTIVE',None)
    return {} if hook is None else hook.require_coverage(16)


def capture_control(worker, command):
    import logits_capture as module
    cap=getattr(module,'_ACTIVE',None)
    if cap is None:
        cap=module.LogitsCapture(worker.get_model()).install();module._ACTIVE=cap
    if command=='finish':return cap.finish()
    if command=='restore':
        cap.restore();module._ACTIVE=None;return {'restored':True}
    cap.begin(baseline=command=='baseline')
    return {'mode':command}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,
            worker_extension_cls='quality_worker.QualityWorkerExtension',
            moe_backend='triton',enable_prefix_caching=False,max_model_len=256,
            max_num_seqs=2,max_num_batched_tokens=256,gpu_memory_utilization=.35,seed=1701)
    prompts=['The purpose of a scientific experiment is to distinguish competing explanations using observations.',
             'A mixture of experts selects several networks for each input token and combines their weighted outputs.']
    arms=[('native',None),('I_bf16',{'mode':'I','precision':'bf16'}),
          ('I_fp8',{'mode':'I','precision':'fp8'}),('I_tiered_h4',{'mode':'I','precision':'tiered','head':4}),
          ('II_bf16',{'mode':'II','precision':'bf16'}),('II_tiered',{'mode':'II','precision':'tiered','head':1})]
    rows=[]
    for name,spec in arms:
        installed=llm.collective_rpc('tiered_set_policy',args=(spec,16))
        llm.collective_rpc('tiered_capture_control',args=('baseline' if name in ['native','II_bf16'] else 'candidate',))
        result=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0),use_tqdm=False)
        samples=[]
        for r in result:
            logps=[float(scores[token].logprob) for token,scores in zip(r.prompt_token_ids,r.prompt_logprobs) if scores is not None]
            samples.append({'prompt_token_ids':r.prompt_token_ids,'true_token_logprobs':logps,'nll_sum':-sum(logps),'tokens':len(logps),'output_token_ids':list(r.outputs[0].token_ids)})
        kl=llm.collective_rpc('tiered_capture_control',args=('finish',))
        diagnostics=llm.collective_rpc('tiered_finish_policy',args=(16,))
        rows.append({'arm':name,'distribution_kl':kl,'installation':installed,'coverage':[d['coverage'] for d in diagnostics],'samples':samples})
        out.write_text(json.dumps({'status':'RUNNING','model':a.model,'scope':'two-prompt integration smoke only','results':rows},indent=2))
    llm.collective_rpc('tiered_set_policy',args=(None,16))
    llm.collective_rpc('tiered_capture_control',args=('restore',))
    for stats in rows[1]['distribution_kl']:
        assert stats['token_kl'] and max(abs(x) for x in stats['token_kl'])<1e-6
    for base,identity in zip(rows[0]['samples'],rows[1]['samples']):
        assert base['prompt_token_ids']==identity['prompt_token_ids']
        assert max(abs(x-y) for x,y in zip(base['true_token_logprobs'],identity['true_token_logprobs']))<1e-5
    out.write_text(json.dumps({'status':'PASS','model':a.model,'scope':'two-prompt integration smoke only; no four-model quality claim','results':rows},indent=2))
    print('PASS')

if __name__=='__main__':main()
