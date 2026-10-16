"""Native reduced-k integration and all-scope numerical comparison.

Use the GPU-lock launcher. The one-prompt comparison assumes routing selection
and actual gate weight have the same ranking (no correction bias). Decode
coverage additionally checks original-k prefill, reduced-k decode, restoration,
and that separate route counting preserves generated token IDs.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--drop-j',type=int,default=4)
    p.add_argument('--counter-backend',choices=['torch','triton','triton_multi'],default='torch')
    p.add_argument('--max-token-kl',type=float,default=1e-3)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    from vllm import LLM,SamplingParams
    llm=LLM(model=args.model,dtype='bfloat16',tensor_parallel_size=1,
            enforce_eager=True,compilation_config={'mode':0,'cudagraph_mode':'NONE'},
            worker_cls='skip_worker.ExpertSkipWorker',moe_backend='triton',
            enable_prefix_caching=False,max_model_len=512,max_num_seqs=8,
            max_num_batched_tokens=512,gpu_memory_utilization=.35,seed=1701)
    rows=[]

    def save(status='RUNNING',**extra):
        out.write_text(json.dumps({'status':status,'rows':rows,**extra},indent=2))

    def run(label,spec,prompts,scope,max_tokens=1,capture=None):
        installed=llm.collective_rpc('skip_set_policy',args=(spec,scope,4096,False))
        if capture:
            llm.collective_rpc('skip_capture_control',args=(capture,))
        result=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=max_tokens,
                           ignore_eos=max_tokens>1,prompt_logprobs=0 if capture else None),use_tqdm=False)
        kl=llm.collective_rpc('skip_capture_control',args=('finish',)) if capture else None
        row={'label':label,'scope':scope,'installation':installed,'kl':kl,
             'worker':llm.collective_rpc('skip_snapshot'),
             'outputs':[{'tokens':list(r.outputs[0].token_ids),'text':r.outputs[0].text}
                        for r in result]}
        rows.append(row)
        save()
        return row

    native={'mode':'global_native','drop_j':args.drop_j,'backend':args.counter_backend,'count_routes':False}
    post={'mode':'global','drop_j':args.drop_j,'mass_budget':1.,'backend':args.counter_backend}
    prompt='The purpose of a scientific experiment is to distinguish competing explanations using observations.'
    try:
        original=run('original_before',None,[prompt],'all')
        reduced=run('global_native_all',native,[prompt],'all',capture='baseline')
        for worker in reduced['worker']:
            for meta in worker['execution'].values():
                if not meta['native_postglobal_equivalence_expected']:
                    raise ValueError('native/post-global equivalence does not apply to biased selection scores')
                assert meta['current_router_top_k']==meta['original_top_k']
            for trace in worker['trace']:
                expected=max(1,trace['original_top_k']-args.drop_j)
                assert trace['executed_top_k']==expected
                assert all(e>=0 for token in trace['post_topk_ids'] for e in token)
        post_row=run('global_post_all',post,[prompt],'all',capture='candidate')
        for metrics in post_row['kl']:
            assert metrics['token_kl'], 'all-scope logit comparison produced no observations'
            assert max(abs(x) for x in metrics['token_kl'])<=args.max_token_kl, metrics
        assert reduced['outputs']==post_row['outputs'], 'native/post-global argmax output changed'
        llm.collective_rpc('skip_capture_control',args=('restore',))
        restored=run('original_restored',None,[prompt],'all')
        assert restored['outputs']==original['outputs']
        for batch in [1,8]:
            prompts=[prompt+f' Give example number {i+1}.' for i in range(batch)]
            uncounted=run(f'global_native_decode_b{batch}',native,prompts,'decode',max_tokens=6)
            counted=run(f'global_native_counted_decode_b{batch}',{**native,'count_routes':True},
                        prompts,'decode',max_tokens=6)
            assert uncounted['outputs']==counted['outputs'], 'route counters changed generated token IDs'
            for worker in counted['worker']:
                assert worker['phase_steps'].get('decode',0)>0
                assert worker['layers']['logical_route_counters_enabled']
                assert worker['layers']['totals']['removed_gate_mass_sum'] is None
                assert worker['layers']['totals']['events']>0
                phases=set()
                for trace in worker['trace']:
                    phases.add(trace['phase'])
                    expected=(max(1,trace['original_top_k']-args.drop_j)
                              if trace['phase']=='decode' else trace['original_top_k'])
                    assert trace['executed_top_k']==expected,trace
                    assert trace['forward_completed']
                assert {'prefill','decode'}<=phases,phases
                assert all(meta['current_router_top_k']==meta['original_top_k']
                           for meta in worker['execution'].values())
            disabled=run(f'disabled_decode_b{batch}',None,prompts,'decode',max_tokens=2)
            for worker in disabled['worker']:
                assert all(t['executed_top_k']==t['original_top_k'] for t in worker['trace'])
        save('PASS',completed_unix=time.time(),max_token_kl_tolerance=args.max_token_kl,
             scope='integration and native/post-global numerical equivalence; not task quality or timed service performance')
        print('PASS',flush=True)
    except Exception as exc:
        save('FAILED',error=f'{type(exc).__name__}: {exc}',failed_unix=time.time())
        raise


if __name__=='__main__':
    main()
