"""Bounded CPU allocation certificate from before1026; no GPU policy prediction."""
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from service_window_model import growth


def extract(before, aliases, config):
    target = 'memory-train-article-0020902'
    def request(rid):
        q = before['requests'][rid]
        return dict(id=aliases[rid], computed=q['computed_tokens'],
            history=q['prompt_tokens']+q['output_tokens'], held=sum(q['block_counts']),
            outputs=q['output_tokens'], output_limit=config['output_tokens'])
    rows = [request(rid) for rid in before['running_ids']]
    t = request(next(rid for rid in before['requests'] if aliases.get(rid)==target))
    assert t['computed']==t['held']==0 and t['history']==3772
    assert all(p['computed']==p['history']-1 and p['outputs']<p['output_limit'] for p in rows)
    assert sum(sum(q['block_counts']) for q in before['requests'].values())==before['pool']['used_blocks']
    return dict(target=t, peers=rows, free=before['pool']['free_blocks'], block=16,
        token_budget=1024, capacity=before['pool']['usable_blocks'],known_model_context=4096)


def arithmetic(state, prefix):
    """At most7 batches; admission fits current history, cap frees only after return."""
    peers, t, block = state['peers'][:prefix], state['target'], state['block']
    added={p['id']:0 for p in peers}; retired=set(); rows=[]; computed=t['computed']; produced=0
    peak=0
    for batch in range(1,8):
        active=[p for p in peers if p['id'] not in retired]
        for p in active: added[p['id']]+=1
        remaining=t['history']-computed if produced==0 else 1
        execute=min(state['token_budget']-len(active),remaining)
        computed+=execute
        assert computed<=state['known_model_context']
        assert all(p['outputs']+added[p['id']]<=p['output_limit'] and
            p['history']+added[p['id']]-1<=state['known_model_context'] for p in active)
        next_outputs=produced+int(computed>=t['history'])
        peer_blocks={p['id']:p['held']+growth(SimpleNamespace(history_tokens=p['history'],
            gpu_blocks=p['held']),added[p['id']],block) for p in active}
        peer_delta=sum(peer_blocks[p['id']]-p['held'] for p in active)
        released_initial=sum(p['held'] for p in peers if p['id'] in retired)
        target_delta=max(0,ceil(computed/block)-t['held'])
        demand=peer_delta+target_delta-released_initial
        admission_ok=batch!=1 or state['free']-peer_delta>=ceil(t['history']/block)-t['held']
        feasible=demand<=state['free'] and admission_ok
        finished=[p['id'] for p in active if p['outputs']+added[p['id']]==p['output_limit']]
        peak=max(peak,demand)
        rows.append(dict(batch=batch, hypothetical_step=1025+batch, target_execute_positions=execute,
            target_computed= computed, target_output_opportunities=next_outputs,
            peer_outputs_this_batch=len(active), cumulative_peer_growth_blocks=peer_delta,
            initial_peer_blocks_released_before_batch=released_initial,target_new_blocks=target_delta,
            joint_new_blocks_before_return=demand, free_after_allocations=state['free']-demand,
            block_shortfall=max(0,demand-state['free']),current_history_admission_ok=admission_ok,
            feasible_before_return=feasible, cap_finished_if_batch_executes=finished,
            cap_blocks_release_only_after_return=sum(peer_blocks[rid] for rid in finished)))
        if not feasible: break
        produced=next_outputs;retired.update(finished)
        if produced>=4: break
    return dict(peer_prefix=prefix, rows=rows, first_output_reached=any(
        r['feasible_before_return'] and r['target_output_opportunities']>=1 for r in rows),
        successful_target_outputs=produced, peak_joint_new_blocks=peak,
        paused_peers=[dict(request_id=p['id'],retained_blocks=p['held'],
            foregone_opportunities_to_first_output=4,foregone_opportunities_in_seven_batches=7)
            for p in state['peers'][prefix:]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    assert not args.output_dir.exists()
    raw=json.loads(args.raw.read_text());config=json.loads((args.raw.parent/'config.json').read_text())
    # Only this object, static limits and ID aliases reach the candidate arithmetic.
    state=extract(raw['memory_trace'][1026]['before'],raw['internal_to_source'],config)
    all_peers=arithmetic(state,len(state['peers']))
    candidates=[arithmetic(state,k) for k in range(len(state['peers'])-1,-1,-1)]
    prefix=next(c for c in candidates if c['first_output_reached'])
    # Future receipt is read only after all candidate rows have been constructed.
    failures=raw['memory_trace'][1029]['allocation_failures']
    validation=[dict(request_id=raw['internal_to_source'][f['internal_request_id']],
        requested_positions=f['requested_tokens'],computed=f['computed_tokens'],
        free_blocks=f['before']['free_blocks']) for f in failures]
    result=dict(status='POSTHOC_PRESTATE_STRUCTURAL_CERTIFICATE',causal_cutoff='memory_trace[1026].before',
        input_state=state,all_peers=all_peers,longest_feasible_fcfs_prefix=prefix,
        prefix_rule='Longest native-FCFS prefix that reaches first output; target always retained. Paused peers retain all KV; no victim search or release.',
        assumptions=['No new arrivals or unknown EOS; one output per active peer per batch.',
            'Known output-cap completions release KV only after a feasible batch returns.',
            'At most7 hypothetical batches and target4 new-output opportunities; no runtime beyond output/context caps.',
            'Fixed physical blocks and1024 token budget; no latency, kernel-cost or amortization claim.'],
        validation_only_1029=validation,source_raw=str(args.raw),
        source_raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        growth_source=str(Path(__file__).with_name('service_window_model.py')),
        growth_source_sha256=hashlib.sha256(Path(__file__).with_name('service_window_model.py').read_bytes()).hexdigest(),
        boundary='This is an observed-state conditional allocation certificate, not prospective GPU action ranking or an alternative executed trajectory. All-peer failure is local, not workload infeasibility.')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'certificate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_peers=all_peers['rows'],prefix=prefix['peer_prefix'],
        prefix_rows=prefix['rows'],paused=prefix['paused_peers'],validation=validation)))


if __name__=='__main__':main()
