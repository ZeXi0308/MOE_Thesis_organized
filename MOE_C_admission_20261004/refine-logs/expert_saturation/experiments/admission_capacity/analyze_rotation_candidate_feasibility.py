#!/usr/bin/env python3
"""Read recorded pre-action states; diagnose funding, never predict outcomes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=h.hexdigest())


def blocks(state):
    values = state['block_counts']
    require(len(values) == 1 and type(values[0]) is int and values[0] >= 0,
            'missing or unsupported per-request block count')
    return values[0]


def classify(rows, chosen):
    by_id = {r['internal_id']: r for r in rows}
    require(chosen in by_id, 'ranked victim is not eligible')
    selected = by_id[chosen]
    return dict(selected_can_fund=selected['funding_slack_blocks'] >= 0,
                eligible_count=len(rows),
                feasible_count=sum(r['funding_slack_blocks'] >= 0 for r in rows),
                rank_first_missed=selected['funding_slack_blocks'] < 0
                and any(r['funding_slack_blocks'] >= 0 for r in rows))


def verify_release(event, state, released):
    require(event['original_preemption_called'] and event['original_preemption_returned'],
            'native preemption did not complete')
    require(event['victim_state'] == state, 'victim changed before native release')
    require(blocks(event['victim_state_after']) == 0, 'victim blocks not released')
    require(event['pool_after']['free_blocks'] - event['pool']['free_blocks'] == released,
            'actual free delta differs from predicted release')


def negative_controls():
    rows = [dict(internal_id='ranked', funding_slack_blocks=-1),
            dict(internal_id='alternative', funding_slack_blocks=1)]
    require(classify(rows, 'ranked')['rank_first_missed'], 'missed-action control failed')
    require(not classify(rows, 'alternative')['rank_first_missed'], 'feasible control failed')
    rejected = []
    for name, callback in [('missing_blocks', lambda: blocks({})),
                           ('absent_selected', lambda: classify(rows, 'absent'))]:
        try:
            callback()
        except (KeyError, ValueError):
            rejected.append(name)
    require(len(rejected) == 2, 'omission controls did not reject')
    return dict(status='PASS_CPU_FIXTURES_ONLY', missed_action_detected=True,
                feasible_action_not_flagged=True, omissions_rejected=rejected)


def analyze_cell(base):
    raw_path = next((p for p in (base / 'raw.json', base / 'raw.json.gz') if p.exists()), None)
    require(raw_path is not None, 'raw missing')
    paths = [raw_path, base / 'headroom-decisions.json', base / 'config.json',
             base / 'engine_args.json', base / 'safe-cap-qualification.json']
    raw, decisions, config, args, qualification = map(read, paths)
    require(args['enable_prefix_caching'] is False and qualification['prefix_caching'] is False,
            'shared cached-block release is outside this diagnostic')
    require(qualification['group_count'] == 1 and qualification['watermark_blocks'] == 0
            and qualification['coordinator_type'] == 'KVCacheCoordinatorNoPrefixCache'
            and qualification['spec_type'] == 'FullAttentionSpec'
            and not qualification['use_eagle'] and qualification['num_lookahead_tokens'] == 0,
            'unsupported block semantics')
    block_size, cfg = qualification['block_size'], config['rotation_config']
    memory, schedule = raw['memory_trace'], raw['scheduler_steps']
    aliases = raw['internal_to_source']
    events = {(e['attempted_step'], e['victim_internal_request_id']): e
              for e in raw['preemption_events']}
    require(len(memory) == len(schedule) == len(decisions), 'step counts differ')
    require(len(events) == len(raw['preemption_events']), 'duplicate preemption event')
    absent, counts, resident = {}, Counter(), {}
    cases, noop = [], Counter()
    before_checks = proposal_checks = release_checks = 0
    for index, (m, s, d) in enumerate(zip(memory, schedule, decisions)):
        require(m['attempted_step'] == s['step'] == d['step'] == index, 'step alignment differs')
        before, after = m['before'], m['after']
        states, running = before['requests'], before['running_ids']
        free = before['pool']['free_blocks']
        require(free == d['free_before'], 'decision and pre-state free blocks differ')
        require(sum(blocks(x) for x in states.values()) == before['pool']['used_blocks'],
                'pre-state block ownership does not close with pool')
        require(before['pool']['used_blocks'] + free == qualification['usable_blocks'],
                'pool conservation failed')
        for rid, state in states.items():
            require(state['num_preemptions'] == counts[rid], 'preemption history differs')
        before_checks += 1
        proposal = d.get('proposal')
        if proposal:
            proposal_checks += 1
            require(all(states[r]['output_tokens'] > 0 and states[r]['computed_tokens']
                        == states[r]['prompt_tokens'] + states[r]['output_tokens'] - 1
                        for r in running), 'selector called outside pure decode state')
            noop[proposal['reason']] += 1
        if proposal and proposal['action'] == 'rotate':
            target, chosen = proposal['resume_id'], proposal['victim_id']
            require(target in states and target not in running and target in absent,
                    'target lacks observed waiting/preemption state')
            target_state = states[target]
            need = max(0, (target_state['prompt_tokens'] + target_state['output_tokens']
                          + block_size - 1) // block_size - blocks(target_state))
            require(need == d['candidate_required_blocks'], 'full-history requirement differs')
            require(index - absent[target] == proposal['absence_steps'], 'absence differs')
            rows = []
            for rid in running:
                state = states[rid]
                progress = min(1., max(0, state['computed_tokens'] - state['prompt_tokens'])
                               / max(1, config['output_tokens']))
                if (progress >= cfg['protect_progress_fraction']
                        or counts[rid] >= cfg['max_absences_per_request']
                        or index - resident.get(rid, -10**9) < cfg['min_residency_steps']):
                    continue
                owned = blocks(state)
                rows.append(dict(internal_id=rid, request_id=aliases[rid],
                                 computed_tokens=state['computed_tokens'],
                                 output_tokens=state['output_tokens'], progress=progress,
                                 allocated_blocks=owned, funding_slack_blocks=free + owned - need))
            least = min(rows, key=lambda r: (r['progress'], r['internal_id']))['internal_id']
            most = min(rows, key=lambda r: (-r['output_tokens'], r['internal_id']))['internal_id']
            # Older fixed-order captures store this only in frozen config.
            order = d.get('effective_victim_order', config.get('rotation_victim_order'))
            require(order in ('least_progress', 'most_output'), 'unknown effective rank')
            require(chosen == (least if order == 'least_progress' else most), 'ranking differs')
            selected_blocks = blocks(states[chosen])
            require(selected_blocks == d['candidate_released_blocks'], 'candidate release differs')
            result = classify(rows, chosen)
            applied = chosen in d['forced_preempted']
            require(applied == result['selected_can_fund'], 'funding and actual application differ')
            actual_release = None
            if applied:
                event = events[index, chosen]
                verify_release(event, states[chosen], selected_blocks)
                require(event['pool']['free_blocks'] == free, 'release does not start at pre-state pool')
                actual_release = event['pool_after']['free_blocks'] - free
                require(target in d['resumed'], 'funded target did not resume in this step')
                release_checks += 1
            cases.append(dict(step=index, free_blocks=free, required_blocks=need,
                              target=aliases[target], victim=aliases[chosen], victim_order=order,
                              actual_released_blocks=actual_release, applied=applied, **result,
                              most_output=aliases[most], least_progress=aliases[least],
                              min_candidate_slack_blocks=min(r['funding_slack_blocks'] for r in rows),
                              max_candidate_slack_blocks=max(r['funding_slack_blocks'] for r in rows),
                              candidates=rows))
        for rid in d['preempted']:
            absent.setdefault(rid, index)
            counts[rid] += 1
            resident.pop(rid, None)
        for rid in d['resumed']:
            absent.pop(rid, None)
            resident[rid] = index
        require(after['pool']['free_blocks'] == d['free_after'], 'after-state pool differs')
    return dict(label=base.name, path=str(base.resolve()), prefix_caching=False,
                evidence_type='CPU_OBSERVED_PRE_ACTION_FEASIBILITY_ONLY',
                inputs=list(map(fingerprint, paths)),
                summary=dict(pre_state_pool_checks=before_checks, selector_calls=proposal_checks,
                             proposed_rotations=len(cases), actual_release_checks=release_checks,
                             candidate_rows=sum(c['eligible_count'] for c in cases),
                             feasible_candidate_rows=sum(c['feasible_count'] for c in cases),
                             rank_first_missed=sum(c['rank_first_missed'] for c in cases),
                             selected_unfunded=sum(not c['selected_can_fund'] for c in cases),
                             rank_choices_differ=sum(c['most_output'] != c['least_progress'] for c in cases),
                             min_candidate_slack_blocks=min(c['min_candidate_slack_blocks'] for c in cases)),
                proposal_reasons=dict(noop), cases=cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    require(not args.output_dir.exists(), 'output directory exists; never overwrite analysis')
    result = dict(question='Does rank-first victim selection miss a fundable full-history recovery '
                           'in the recorded pre-action states?',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  evidence_ceiling='CPU_OBSERVED_PRE_ACTION_FEASIBILITY_ONLY',
                  command=sys.argv, script=fingerprint(Path(__file__)),
                  controls=negative_controls(), cells=[])
    for cell in args.cell:
        result['cells'].append(analyze_cell(cell))
    misses = sum(c['summary']['rank_first_missed'] for c in result['cells'])
    result['verdict'] = 'OBSERVED_MISSED_FUNDED_ACTION' if misses else 'NO_OBSERVED_RANK_FIRST_FUNDING_MISS'
    result['limitations'] = [
        'Only recorded proposal states on their own policy trajectories; repeats are correlated.',
        'Unselected releases use observed block counts under qualified unshared full-attention semantics; '
        'only selected victims have a directly measured native free delta.',
        'Candidate construction reads before snapshots and preceding events only; same-step events '
        'validate the executed action, never provide candidate scores.',
        'No policy-specific alternative future state, latency ranking, Oracle, quality or method gain.',
        'A funded full history does not alone guarantee completion; the runtime also protects growth.',
        'No prefix-cache sharing, heterogeneous/EOS termination or different pressure regime is tested.']
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'analysis.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    lines = ['# 当前前态中的轮转资助资格', '', '新增问题：先排序再检查资源是否漏掉可执行交换？', '',
             f"Verdict: `{result['verdict']}`；CPU 只读诊断，无新 GPU 执行。", '',
             '| Cell | 前态核对 | 交换提案/实际释放核对 | 合法/可资助候选 | 漏动作 | 最小余量块 |',
             '|---|---:|---:|---:|---:|---:|']
    for c in result['cells']:
        s = c['summary']
        lines.append(f"| {c['label']} | {s['pre_state_pool_checks']} | "
                     f"{s['proposed_rotations']}/{s['actual_release_checks']} | "
                     f"{s['candidate_rows']}/{s['feasible_candidate_rows']} | "
                     f"{s['rank_first_missed']} | {s['min_candidate_slack_blocks']} |")
    lines += ['', '表中候选为 request × 已观察提案前态，不是独立请求或实验重复；'
              '完整枚举、身份、块数与输入 SHA256 见 `analysis.json`。', '',
              '块守恒：free + sum(owned) = usable；恢复需求 '
              '`ceil((prompt + observed_output)/block_size) - target_owned`。'
              '候选保留进度、驱逐次数和最短驻留约束；仅检查 `free + victim_owned >= need`。', '',
              '实际选中者由原生抢占前后 pool 差额核对；未选候选只得到结构上的可资助性，'
              '没有执行其未来状态。合成负控检验“第一名不足、另一名足够”会被检测；'
              '缺块字段或被选者不在候选集中会拒绝分析。', '',
              '当前三个 cell 均关闭 prefix caching；结果不覆盖共享缓存块、'
              '自然 EOS、异构终止或其它压力点。未测性能/质量，没有 Oracle 或方法 GO。', '',
              '唯一下一步：继续原冻结同路径运行时重复；本域若无遗漏，不以调整资助排序新增 GPU 实验。', '',
              '重跑必须使用新的输出目录：', '', '```text',
              'python3 -B ' + ' '.join(sys.argv), '```', '']
    with (args.output_dir / 'REPORT.md').open('x') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(verdict=result['verdict'], cells=[dict(label=c['label'], **c['summary'])
                                                          for c in result['cells']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
