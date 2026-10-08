#!/usr/bin/env python3
"""One fixed step-405 prestate; prediction inputs are separated from later checks."""
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

TARGET = 'memory-train-article-0003640'
VICTIM = 'memory-train-article-0003571'
STEP, BLOCK, EXTRA = 405, 16, 3


def predict(before, target, victim, released, budget, chunk, aliases, arrivals):
    """No later state, output event, EOS, completion, or timing enters this function."""
    free = before['pool']['free_blocks'] + released
    t = before['requests'][target]
    history = t['prompt_tokens'] + t['output_tokens']
    remaining = history - t['computed_tokens']
    peers = {r: before['requests'][r] for r in before['running_ids'] if r != victim}
    assert target not in peers and remaining > 0
    assert all(v['prompt_tokens'] + v['output_tokens'] - v['computed_tokens'] == 1
               for v in peers.values())
    quota = budget - len(peers)
    if chunk:
        quota = min(quota, chunk)
    assert quota > 0
    rounds = ceil(remaining / quota)
    t_initial = t['block_counts'][0]
    reserve_target = max(t_initial, ceil(history / BLOCK)) - t_initial
    rows = []
    for n in range(1, rounds + EXTRA + 1):
        computed = min(history, t['computed_tokens'] + n * quota) if n <= rounds else history + n - rounds
        peer_growth = {r: max(0, ceil((v['prompt_tokens'] + v['output_tokens'] + n - 1) / BLOCK)
                             - v['block_counts'][0]) for r, v in peers.items()}
        actual_target = max(t_initial, ceil(computed / BLOCK)) - t_initial
        physical = actual_target + sum(peer_growth.values())
        reserved = max(reserve_target, actual_target) + sum(peer_growth.values())
        target_tokens = min(quota, remaining - (n-1)*quota) if n <= rounds else 1
        rows.append(dict(opportunity_round=n, phase='restore' if n <= rounds else 'post_first_output',
            target_execution_tokens=target_tokens, peer_execution_tokens=len(peers),
            target_outputs=0 if n < rounds else 1+n-rounds, peer_outputs_each=n,
            physical_increment_blocks=physical, current_history_reservation_plus_peer_growth=reserved,
            remaining_unallocated_history_reservation=max(0, reserve_target-actual_target),
            physical_free=free-physical, reservation_slack=free-reserved,
            token_budget_ok=target_tokens+len(peers)<=budget,
            resource_ok=reserved<=free,
            growing_peers={aliases[r]: g for r, g in peer_growth.items() if g}))
    order = sorted(peers, key=lambda r: (arrivals[aliases[r]], aliases[r]))
    peer_blocks_at_first = {r: max(v['block_counts'][0], ceil((v['prompt_tokens']+
        v['output_tokens']+rounds-1)/BLOCK)) for r, v in peers.items()}
    slots = {r: BLOCK*peer_blocks_at_first[r] - (peers[r]['prompt_tokens']+
        peers[r]['output_tokens']+rounds-1) for r in peers}
    fit = [dict(extra_output_opportunity=j,
        fcfs_peers=[aliases[r] for r in order if slots[r]>=j],
        paused_peers=[aliases[r] for r in order if slots[r]<j],
        target_tokens=1, total_tokens=1+sum(slots[r]>=j for r in peers),
        target_additional_blocks=max(0,ceil((history+j)/BLOCK)-ceil(history/BLOCK)),
        peer_additional_blocks=0) for j in range(1, EXTRA+1)]
    return dict(free_after_fixed_victim=free, target_history=history,
        target_computed=t['computed_tokens'], target_execution_remaining=remaining,
        target_full_history_reservation=reserve_target, peer_count=len(peers),
        restore_token_quota=quota, restore_rounds_lower_bound=rounds,
        rows=rows, fit_after_first_output=fit,
        target_current_history_slack_after_first=BLOCK*ceil(history/BLOCK)-history,
        deferred_peers=[dict(request=aliases[r], additional_outputs_before_pause=slots[r],
            extra_no_service_within_suffix=f'<= {EXTRA-slots[r]} * t_batch')
            for r in order if slots[r]<EXTRA])


def candidate_release(before, victim, args, qualification):
    """Pre-action ownership amount, only in the pinned unshared full-attention domain."""
    expected = dict(group_count=1, manager_group_count=1, prefix_caching=False,
        coordinator_type='KVCacheCoordinatorNoPrefixCache', spec_type='FullAttentionSpec',
        block_size=BLOCK, single_type_block_size=BLOCK, scheduler_block_size=BLOCK,
        dcp_world_size=1, pcp_world_size=1, num_lookahead_tokens=0, num_spec_tokens=0,
        use_eagle=False, watermark_blocks=0)
    assert args['enable_prefix_caching'] is False
    assert all(qualification[k] == value for k,value in expected.items())
    assert victim in before['running_ids']
    assert all(len(v['block_counts']) == 1 for v in before['requests'].values())
    held_sum = sum(v['block_counts'][0] for v in before['requests'].values())
    assert held_sum == before['pool']['used_blocks']
    assert before['pool']['used_blocks'] + before['pool']['free_blocks'] == before['pool']['usable_blocks']
    assert before['pool']['usable_blocks'] == qualification['usable_blocks']
    return dict(candidate_release_blocks=before['requests'][victim]['block_counts'][0],
        source='before.requests[fixed_victim].block_counts[0]',
        pre_action_held_sum=held_sum, pre_action_pool_used=before['pool']['used_blocks'],
        qualification=expected,
        capability='Pinned install rejects caching/shared coordinator, multiple KV groups, connectors, deferred free, speculative/parallel cache and non-FullAttentionManager. Qualification plus before ownership census permits held==candidate releasable only in this regime.',
        boundary='No native receipt or pool_after enters the candidate amount. APC-off and held-sum equality alone are not a universal proof for other managers/backends.')


def inspect(folder):
    path = folder/'raw.json'; blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest(); raw = json.loads(blob); del blob
    aliases = raw['internal_to_source']
    target = next(r for r, s in aliases.items() if s == TARGET)
    victim = next(r for r, s in aliases.items() if s == VICTIM)
    before = next(m['before'] for m in raw['memory_trace'] if m['attempted_step']==STEP)
    args = json.loads((folder/'engine_args.json').read_text())
    qualification = json.loads((folder/'safe-cap-qualification.json').read_text())
    candidate = candidate_release(before, victim, args, qualification)
    # Frozen capability checks are known before execution, not release receipts.
    environment = json.loads((folder/'environment.json').read_text())
    package = folder.parents[1]/'pkg'
    capabilities = {}
    for name in ('ltr_recompute_native.py', 'run_probe.py'):
        body = (package/name).read_bytes()
        source_sha = hashlib.sha256(body).hexdigest()
        assert source_sha == environment['source_sha256'][name]
        capabilities[name] = source_sha
    candidate['capability_sources_sha256'] = capabilities
    chunk = json.loads((folder/'resolved-scheduler-config.json').read_text())['long_prefill_token_threshold']
    assert not args['enable_prefix_caching'] and args['max_num_batched_tokens']==1024
    arrivals = {r['request_id']:r['arrival_s'] for r in raw['requests']}
    prediction = predict(before,target,victim,candidate['candidate_release_blocks'],
                         args['max_num_batched_tokens'],chunk,aliases,arrivals)
    # Input cutoff ends above. Receipts and later calls are validation only.
    receipts = [e for e in raw['preemption_events'] if e['attempted_step']==STEP]
    assert len(receipts)==1 and receipts[0]['victim_internal_request_id']==victim
    receipt = receipts[0]
    assert receipt['pool']==before['pool'] and receipt['original_preemption_returned']
    observed_release = receipt['pool_after']['free_blocks']-receipt['pool']['free_blocks']
    release_matches = observed_release == candidate['candidate_release_blocks']
    assert release_matches
    checks = []
    for row in prediction['rows'][:prediction['restore_rounds_lower_bound']]:
        step = STEP+row['opportunity_round']-1
        schedule = next(s for s in raw['scheduler_steps'] if s['step']==step)
        actual_target = next(s['scheduled_tokens'] for s in schedule['scheduled'] if s['internal_request_id']==target)
        actual_peer = sum(s['scheduled_tokens'] for s in schedule['scheduled'] if s['internal_request_id']!=target)
        memory = next(m for m in raw['memory_trace'] if m['attempted_step']==step)
        ok = (actual_target==row['target_execution_tokens'] and actual_peer==row['peer_execution_tokens']
              and memory['after']['pool']['free_blocks']==row['physical_free'])
        checks.append(dict(step=step, actual_target_tokens=actual_target,actual_peer_tokens=actual_peer,
            actual_free_after=memory['after']['pool']['free_blocks'],matches_conditioned_all_peer_prefix=ok))
    output_count = before['requests'][target]['output_tokens']+1
    first = next(e for e in raw['output_events'] if e['request_id']==TARGET and e['cumulative_tokens']==output_count)
    next_step = STEP+prediction['restore_rounds_lower_bound']
    next_schedule = next(s for s in raw['scheduler_steps'] if s['step']==next_step)
    assert first['received_s']<=next_schedule['start_s']
    cutoff = next(s['start_s'] for s in raw['scheduler_steps'] if s['step']==STEP)
    last = {}
    for e in raw['output_events']:
        if e['received_s']<=cutoff and e['chunk_size']>0:
            last[e['request_id']]=e['received_s']
    ages = {rid:cutoff-last[rid] for rid in (TARGET,VICTIM,
        *[p['request'] for p in prediction['deferred_peers']])}
    return dict(label=folder.name,raw_path=str(path),raw_sha256=digest,
        input_before_step=STEP, candidate_release_from_before=candidate,
        before_state={aliases[r]:v for r,v in before['requests'].items()},
        output_ages_at_schedule_entry_s=ages,prediction=prediction,
        validation_only=dict(receipt_released_blocks=observed_release,
            receipt_matches_before_candidate=release_matches,
            prefix=checks,first_new_output_count=output_count,
            first_output_engine_return_s=first['received_s'],
            next_step_actual_victims=next_schedule['preempted_request_ids'],
            no_alternative_future_was_executed=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    jp=args.output_dir/'pre_resume_window_certificate_v2.json';mp=args.output_dir/'pre_resume_window_certificate_v2.md'
    if jp.exists() or mp.exists(): raise SystemExit('preserve existing certificates; use a new output directory')
    cells=[inspect(args.results_root/label) for label in ('block0-guard_residual','block1-guard_residual')]
    result=dict(status='STRUCTURAL_PRE_RESUME_ALL_PEER_WINDOW_CONDITIONALLY_INFEASIBLE',
        selection='Fixed target3640, step405, fixed victim3571; no alternate target/victim search.',
        input_source_correction='v1 passed a release value derived from pool_after to predict. v2 derives candidate212 only from before held under pinned no-sharing capability and census checks; receipt212 is validation_only. Numerical conclusion unchanged; v1 files retained.',
        assumptions='Current 30 peers remain active and receive one new output per recovery/suffix batch; no unknown EOS/free, new arrival, other restore, or extra victim is credited. Synchronous one-output decode; fixed 1024 token budget, chunk threshold0 and block16.',
        interpretation='Restore-round count is a token-budget lower bound, attained only if the proposed all-peer allocations execute. More small chunks require more peer growth if peers continue each batch; early terminals or changed peer service require a separate qualification.',
        tax_boundary='3274 covers 3273 historical positions plus the final pending input that yields a new output. It is execution coverage, not pure recompute tax or milliseconds.',
        next_gpu='Reuse already selected cohort3 native/most/fit/residual eight cells; no new window GPU run.',cells=cells)
    args.output_dir.mkdir(parents=True,exist_ok=True);jp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=cells[0]['prediction']['rows']
    table='\n'.join(f"| {r['opportunity_round']} | {r['target_execution_tokens']} | {r['peer_execution_tokens']} | {r['target_outputs']} | {r['physical_increment_blocks']} | {r['current_history_reservation_plus_peer_growth']} | {r['reservation_slack']} |" for r in rows)
    md='''# 恢复之前的服务窗口证书 v2：固定 step405

Verdict：`STRUCTURAL_PRE_RESUME_ALL_PEER_WINDOW_CONDITIONALLY_INFEASIBLE`。在405真正动作前就能发现：固定原victim释放资源后，恢复到首输出可以兑现，但若全部30个当前peer每批继续输出，再给target三个输出机会会缺6块。该结论不需要先看到409；它也不意味着fit子集或整个窗口无解。

两repeat前态相同：free=0；固定victim3571在405 before持有212块；target3640的history=3274、computed=0、held=0，完整当前历史需要205块。固定victim移出后30个resident均pending1。

输入来源纠正：v1虽然校验了receipt释放量等于before持有量，仍将从pool_after得到的212传入predict，因此其“完全pre-action输入”表述不成立。v2的212直接取before victim held，原生receipt只进入validation_only。v1脚本及JSON/MD全部保留，数值结论未变。

候选释放量的能力资格来自执行前配置和已冻结安装检查：APC关闭、唯一KV group、FullAttentionSpec/FullAttentionManager、KVCacheCoordinatorNoPrefixCache、无KV connector/延迟释放/speculative或并行cache；safe-cap-qualification与已执行源hash一致。405 before所有请求均只有一个block_counts分量，Σheld=used=6656，free+used=usable=6656。只有这个已明确的不共享后端和完整前态资源账本允许把固定victim持有的212作为候选可释放量；不能仅凭APC off或sum相等推广到其它后端。原生receipt之后独立确认释放212，不参与预测输入。

预算1024、实际chunk threshold=0。每个peer每批1token，则target每批至多994；恢复剩余3274个输入位置至少要4批，最大分配路径为994/994/994/292。3274包含3273个已经执行过的历史位置和首输出的最后一个输入，不全是重算税，不能换算为毫秒。未读取未来EOS或未来分配来推导这4批。

| 条件机会批 | target执行位置 | peer执行位置 | target恢复后累计新输出 | 累计实际新增块 | 完整当前历史保留+peer增长 | 在212块内余量 |
|---:|---:|---:|---:|---:|---:|---:|
'''+table+'''

前4批资源可容纳；第4批首输出时205个target块加peer新增7块正好用满212块。第5/6/7批如果all-peer继续共同输出，分别再缺3/5/6块。若只看恢复时205≤212并忽略peer增长，会错误把7块余量当作恢复后仍可用；若连恢复后的增长也忽略，就会签出无法兑现的四输出承诺。

完整历史保留与物理已分配量分列：未分配的历史余量只算一次，不和完整历史重复相加。这里的保留是资源资格，不要求立即物理分配全部未来KV。

## 同一前态推导的fit子集与等待代价

用405的peer历史加条件恢复期4个新输出，可直接推导首输出时各自held块边界，随后固定FCFS逐步跳过需要新块的peer：三个额外机会批分别仍可服务28/26/25个请求（含target），新块均0。799/2820/3345从第1个额外批暂停，133/2038从第2批，1401从第3批；窗口内新增无服务等待分别≤3/2/1·t_batch。这不是旧409前态被喂回模型，计算只来自405历史和明确的每批服务假设。

原victim3571在整个条件窗口继续等待，其额外等待需记为恢复4批的总跨度再加3个后续batch跨度，不能只计peer暂停。target自身到首输出还有恢复队列/执行跨度，均未预测毫秒。每个请求在405 schedule入口的当前output age写入JSON，只有当时已经返回的新输出用于age。

符号t_batch是后续三批各自耗时的上界，不是测得的未来毫秒；没有这样的上界就保留各批跨度之和。窗口结束后谁释放资源、被暂停peer何时真正得到输出、target是否再次丢弃状态都仍需资格化，不能把三批内等待界当成完整下一输出间隔保证。

## 条件与独立验证

4批是由当前待处理位置和token预算给出的下界，并非普遍时间预测。只有固定peer集合每批各1token、target拿满剩余预算时才达到。更小chunk或不同优先级可能增加批数；若peers仍继续执行，增长只会更大。未知EOS、完成释放、改变peer集合或其它恢复动作可能改变需求，必须重算，不能把当前条件拒绝扩成全局无解。

预测函数predict只接收405 before、从before持有量得到且通过能力资格的固定victim候选释放量、当前配置及到达顺序；原生receipt和真实后续独立放在validation_only：两个repeat原动作405–408的target/peer执行量及物理free均与上述前4批条件预测匹配，实际首输出累计203；之后409原动作抢占target。这只检验原动作的已执行前缀，不是替代窗口收益或未来Oracle。

当前能回答的是：恢复前已可检测all-peer四输出窗口缺块，同时保留有明确暂停代价的fit子集动作。不能直接据此延长保护，更不能称方法GO。唯一下一GPU仍复用共享cohort3 native/most/fit/residual反序八格，先检验强简单基线是否覆盖残差；窗口动作保持条件接续。

复跑脚本接受--results-root和--output-dir；已存在结果拒绝覆盖。无GPU、无controller，未修改上一轮原件。
'''
    mp.write_text(md)
    print(json.dumps(dict(status=result['status'],cells=[dict(label=c['label'],
        prefix_matches=all(v['matches_conditioned_all_peer_prefix'] for v in c['validation_only']['prefix']),
        final_slack=c['prediction']['rows'][-1]['reservation_slack']) for c in cells])))


if __name__=='__main__':main()
