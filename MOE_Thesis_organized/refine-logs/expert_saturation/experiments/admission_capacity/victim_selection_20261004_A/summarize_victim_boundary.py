#!/usr/bin/env python3
"""Local, read-only closeout table from frozen raw/request/decision records.

No GPU/runtime imports, network, trace mutation, or new performance experiment.
The fixed list is the decisive online comparisons, not a search for favorable runs.
--details prints per-cell quantities; default prints the one RESULTS.md table.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
BEGIN, END = '<!-- A_BOUNDARY_TABLE_BEGIN -->', '<!-- A_BOUNDARY_TABLE_END -->'
GROUPS = (
    ('remaining-budget / tail', ('remaining-budget-westd53005-20261008-r02',),
     '主指标反序翻转；简单预算有个体收益，无稳定总体优势'),
    ('max-release / remaining-budget', ('max-release-westd53005-20261008-r01',),
     '两对平均flow更差、gap P95更好；更少抢占不等于更快完成'),
    ('cap-bucket / remaining-budget', ('cacheopt-cap-bucket-westd53005-20261009-r01',),
     '主指标翻转、gap P95更差、工作量改变；仅CacheOPT式组件'),
    ('等释放 host once / tail（prefix ON）', ('equal-release-host-once-westd53005-20261008-r02',),
     '真实等释放干预；主指标翻转，替代victim受损'),
    ('partial-restore once / tail（prefix ON）', ('partial-restore-once-westd53005-20261008-r01',),
     '两对平均flow无改善；替代victim再次抢占'),
    ('严格等释放 budget / tail', ('equal-release-budget-westd53005-20261008-r01',),
     '实际零改选；时延波动不是机制效果'),
    ('host-near seeded / tail（4对）', ('host-near-seeded-20261008-r01', 'host-near-seeded-20261008-r02'),
     '3对平均flow变差、1对变好；非等释放、输出序列不同'),
)
ABORTED = ('host-near-20261007-r01', 'host-probe-20261007-r01',
           'host-probe-20261007-r02', 'mixed-budget-spread-20261008-r01',
           'equal-release-host-once-westd53005-20261008-r01',
           'remaining-budget-westd53005-20261008-r01')


def read(path):
    return json.loads(path.read_text())


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * q
    i = int(p)
    return values[i] + (p - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def stats(values):
    return dict(n=len(values), minimum=min(values) if values else None,
                median=median(values) if values else None,
                p95=quantile(values, .95), maximum=max(values) if values else None)


def cell(directory, spec):
    archive = directory / 'archive'
    raw, store, metrics = (read(archive / f) for f in
                           ('raw.json', 'selective-store.json', 'metrics.json'))
    arrived = [r for r in raw['requests'] if r['arrival_s'] <= raw['observation_end_s']]
    counts = Counter(r['status'] for r in arrived)
    complete = counts['completed'] == len(arrived)
    requests, flows, ttfts, gaps, lags = {}, [], [], [], []
    for r in arrived:
        times = sorted(set(r['token_times_s']))
        flow = r['completion_s'] - r['arrival_s'] if r['completion_s'] is not None else None
        ttft = times[0] - r['arrival_s'] if times else None
        gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
        requests[r['request_id']] = dict(flow=flow, ttft=ttft, gap=gap,
            tokens=len(r['output_token_ids']), stop=r.get('stop_reason'),
            sequence=hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest(),
            external=(r['arrival_s'], r['prompt_token_ids_sha256'], r['max_output_tokens']),
            times=times)
        if flow is not None:
            flows.append(flow)
        if ttft is not None:
            ttfts.append(ttft)
        gaps.append(gap)
        lags.append(r['admission_s'] - r['arrival_s'])
    # This prevents a silently truncated cohort from producing an all-request mean.
    assert len(arrived) == metrics['n_arrived'] == spec['requests']
    assert counts['completed'] == metrics['n_completed']
    assert counts['failed'] == metrics['n_failed']
    decisions = store['victim_decisions']
    actual = [e for e in raw['preemption_events'] if
              e.get('original_preemption_called') is True and e.get('original_preemption_returned') is True]
    by_key = defaultdict(list)
    for e in actual:
        by_key[e['engine_call_index'], e['internal_request_id']].append(e)
    changed, release, delta, candidates, waits, unknown_delta = 0, [], [], [], [], 0
    used = set()
    for d in decisions:
        rows = d['candidates']
        # Native suffix legality, NOT the retired `qualified` pure-decode ranking gate.
        indices = [r['index'] for r in rows]
        assert indices == list(range(d['unprocessed_suffix_start'], indices[-1] + 1))
        assert rows[0]['request'] == d['failed_request'] and rows[-1]['request'] == d['native_tail']
        candidates.append(len(rows))
        key = d['step'], d['selected']
        assert len(by_key[key]) == 1 and key not in used
        used.add(key)
        e = by_key[key][0]
        freed = e['actual_released_blocks']
        assert freed == e['free_blocks_after_preempt'] - e['free_blocks_before_preempt']
        selected = next(r for r in rows if r['request'] == d['selected'])
        if selected.get('immediate_releasable_blocks') is not None:
            assert freed == selected['immediate_releasable_blocks']
        release.append(freed)
        is_changed = d['selected'] != d['native_tail']
        assert d['changed'] == is_changed
        changed += is_changed
        if is_changed:
            tail_release = rows[-1].get('immediate_releasable_blocks')
            if tail_release is None:
                unknown_delta += 1  # Never substitute held pages for releasable pages.
            else:
                delta.append(freed - tail_release)
        times = requests[e['request_id']]['times']
        i = bisect_right(times, e['method_returned_s'])
        if i < len(times):
            waits.append(times[i] - e['method_returned_s'])
    assert len(used) == len(actual) == raw['actual_preemption_count']
    victim_counts = Counter(e['request_id'] for e in actual)
    duration = raw['observation_end_s']
    tokens = sum(r['tokens'] for r in requests.values())
    offload_path = archive / 'offload-events.json'
    offload = read(offload_path) if offload_path.exists() else None
    for r in requests.values():
        del r['times']
    return dict(path=str(directory.relative_to(HERE)), rule=spec['native_victim_rule'],
        counts=dict(arrived=len(arrived), completed=counts['completed'], failed=counts['failed'],
                    unfinished=metrics['n_unfinished'], rejected=None, timed_out=None),
        status=raw['status'], prefix_caching=store.get('prefix_caching'),
        flow_mean=mean(flows) if complete else None, flow_p95=quantile(flows, .95) if complete else None,
        ttft_mean=mean(ttfts) if len(ttfts) == len(arrived) else None, ttft_p95=quantile(ttfts, .95),
        gap_p95=quantile(gaps, .95), gap_max=max(gaps), submission_lag_max=max(lags),
        tokens=tokens, token_rate=tokens / duration, request_rate=counts['completed'] / duration,
        duration=duration, drain=duration - max(r['arrival_s'] for r in arrived),
        decisions=len(decisions), candidates=stats(candidates), changed=changed,
        preemptions=len(actual), repeated_victims=sum(n > 1 for n in victim_counts.values()),
        release=stats(release), changed_release_delta=stats(delta), unknown_release_delta=unknown_delta,
        preempt_to_next_output=stats(waits), no_next_output=len(actual) - len(waits),
        selector_s=sum(d['selector_wall_s'] for d in decisions),
        recovery_allocations=len(offload['allocated']) if offload is not None and 'allocated' in offload else None,
        load_acks=len(offload['load_acknowledgements']) if offload is not None and 'load_acknowledgements' in offload else None,
        requests=requests)


def compare(a, b):
    assert a['requests'].keys() == b['requests'].keys()
    for key in a['requests']:
        assert a['requests'][key]['external'] == b['requests'][key]['external']
    delta = {k: b[k] - a[k] if a[k] is not None and b[k] is not None else None
             for k in ('flow_mean', 'ttft_p95', 'gap_p95', 'drain', 'tokens')}
    delta['token_rate_pct'] = 100 * (b['token_rate'] / a['token_rate'] - 1)
    for name, field in (('sequences', 'sequence'), ('lengths', 'tokens'), ('stops', 'stop')):
        delta[name] = sum(a['requests'][k][field] != b['requests'][k][field] for k in a['requests'])
    changes = {k: b['requests'][k]['flow'] - a['requests'][k]['flow']
               for k in a['requests'] if a['requests'][k]['flow'] is not None and b['requests'][k]['flow'] is not None}
    delta['worst_harmed_request'] = max(changes, key=changes.get) if changes else None
    delta['worst_flow_loss'] = max(changes.values()) if changes else None
    return delta


def number(x, signed=False):
    if x is None:
        return '未知'
    return f'{x:+.3f}' if signed else f'{x:g}'


def joined(values, signed=False):
    return '/'.join(number(x, signed) for x in values)


def transition(pairs, field, sub=None):
    def value(c):
        return c[field] if sub is None else c[field][sub]
    vals = [f'{number(value(a))}→{number(value(b))}' for a, b in pairs]
    return vals[0] + '（各对）' if len(set(vals)) == 1 else '/'.join(vals)


def generate():
    lines = [BEGIN,
        '| 完整策略对照 B / A | 合法suffix数；实际非tail改选 A→B | 容量（页） | 再次抢占与停顿 | 全请求 Δ秒：flow均值；TTFT P95；gap P95 | 输出率/排空与工作量 | 当前证据状态 |',
        '|---|---|---|---|---|---|---|']
    details = []
    for title, names, verdict in GROUPS:
        pairs, cells, links = [], [], []
        for name in names:
            session = HERE / ('session-native-' + name)
            plan = read(session / 'plan.json')
            group = [cell(session / f"cell-{i:02d}-{spec['label']}", spec)
                     for i, spec in enumerate(plan['cells'])]
            assert len(group) == 4
            cells.extend(group)
            pairs.extend(((group[0], group[1]), (group[3], group[2])))
            links.append(f'[{len(links) + 1}]({session.name}/)')
        delta = [compare(a, b) for a, b in pairs]
        # Median/range describes candidate states, not a statistical sample of runs.
        nc = []
        for i, label in enumerate(('A', 'B')):
            ds = [pair[i]['candidates'] for pair in pairs]
            medians = sorted({d['median'] for d in ds})
            med = number(medians[0]) if len(medians) == 1 else f'{number(medians[0])}–{number(medians[-1])}'
            nc.append(f"{label} {med}[{number(min(d['minimum'] for d in ds))}–{number(max(d['maximum'] for d in ds))}]")
        legal = '；'.join(nc) + '<br>改选 ' + transition(pairs, 'changed')
        capacity = '释放中位 ' + transition(pairs, 'release', 'median')
        capacity += '<br>B改选Δtail中位 ' + (joined([b['changed_release_delta']['median'] for _, b in pairs])
            if any(b['changed'] for _, b in pairs) else '不适用（零改选）')
        if any(b['unknown_release_delta'] for _, b in pairs):
            capacity += '（未记引用计数）'
        recovery = '抢占 ' + transition(pairs, 'preemptions')
        recovery += '<br>重复victim ' + transition(pairs, 'repeated_victims')
        recovery += '<br>抢占→下次输出P50 ' + '/'.join(
            f"{a['preempt_to_next_output']['median']:.2f}→{b['preempt_to_next_output']['median']:.2f}s" for a, b in pairs)
        effects = '<br>'.join(label + joined([d[k] for d in delta], True) for label, k in
                              (('flow ', 'flow_mean'), ('TTFT ', 'ttft_p95'), ('gap ', 'gap_p95')))
        work = 'token/s ' + joined([d['token_rate_pct'] for d in delta], True) + '%'
        work += '<br>排空 ' + joined([d['drain'] for d in delta], True) + 's'
        work += '<br>Δtokens ' + joined([d['tokens'] for d in delta])
        work += '<br>序列/长度/终止差 ' + '；'.join(f"{d['sequences']}/{d['lengths']}/{d['stops']}" for d in delta)
        done = {tuple(c['counts'][k] for k in ('arrived', 'completed', 'failed', 'unfinished')) for c in cells}
        counts = '各格 到达/完成/失败/未完 ' + '；'.join('/'.join(map(str, x)) for x in sorted(done))
        lines.append('| ' + ' | '.join((title + ' ' + ','.join(links), legal, capacity, recovery,
                                        effects, work, verdict + '<br>' + counts)) + ' |')
        details.append(dict(comparison=title, cells=[{k: v for k, v in c.items() if k != 'requests'} for c in cells],
                            paired_deltas=delta))
    failures = [read(HERE / ('session-native-' + name) / 'receipt.json') for name in ABORTED]
    assert all(r['status'] == 'ABORTED' and r['cells'] == [] for r in failures)
    busy = sum('GPU busy before initialization' in r['error'] for r in failures)
    disk = sum('Insufficient new-host disk space' in r['error'] for r in failures)
    assert busy + disk == len(failures)
    lines.append(f'| 启动失败（历史保留） | 初始化前GPU忙 {busy} 组；磁盘不足 {disk} 组 | 未进入测量 | 无请求轨迹 | 不计入性能对照 | 不能记成请求失败或方法负结果 | receipt均ABORTED、cells为空 |')
    natural = HERE / 'session-native-natural-summary-westd53005-20261009-r01'
    paths = list(natural.glob('cell-*/archive/raw.json'))
    availability = '本地未见原始结果；终态和服务效果未核实' if not paths else '已有本地raw，须单独分析；不计作策略比较'
    lines.append('| 未测/未确认 | 自然摘要：' + availability + ' | 新窗口评分器未执行 | 精确重算/复制成本未隔离 | 应用SLO goodput与独立确认未测 | 输出质量等价未验证 | 完整CacheOPT、第二模型泛化均未证明 |')
    lines.append(END)
    return '\n'.join(lines), details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details', action='store_true', help='Print per-cell metrics as JSON instead of the table.')
    parser.add_argument('--check', action='store_true', help='Compare regenerated table with RESULTS.md; never write.')
    args = parser.parse_args()
    table, details = generate()
    if args.check:
        result = (HERE / 'RESULTS.md').read_text()
        existing = result[result.index(BEGIN):result.index(END) + len(END)]
        if table != existing:
            raise SystemExit('RESULTS.md table differs from raw-data recomputation')
        print('MATCH: RESULTS.md table reproduced from frozen local raw data.')
    else:
        print(json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) if args.details else table)
