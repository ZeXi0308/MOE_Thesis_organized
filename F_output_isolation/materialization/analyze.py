"""Recompute frozen materialization experiments; stdlib, no service execution.

Every run remains a separate repetition. The report describes CPU/delivery
tradeoffs, not capacity, an SLO, statistical significance, or a paper claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ['native1', 'native2', 'native4', 'token4', 'bytes1024',
        'class_static', 'lazy_static']
ORDER = ARMS + list(reversed(ARMS))
SHAS = {'high': 'cb813b551ed43b8aafbfedda87d7f3bba57fbd9cb4e6d993d17d5a63a728a32b',
        'medium': 'f735ed337c4f635337cd954c45335d691ef7970beb0fd36d70f879aed619abc5'}
SEMA = ('text_sha256', 'logprobs_sha256', 'logprobs_count')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(values, p):
    if not values:
        return None
    x = sorted(values)
    i = (len(x) - 1) * p
    lo = int(i)
    return x[lo] * (1 - (i - lo)) + x[min(lo + 1, len(x) - 1)] * (i - lo)


def describe(values):
    return {'n': len(values), 'p50': q(values, .5), 'p95': q(values, .95),
            'p99': q(values, .99), 'max': max(values) if values else None}


def same(actual, expected, label):
    if actual is None or expected is None:
        assert actual is expected, (label, actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), (
            label, actual, expected)


def mapping_for(trace):
    requests = {r['request_id']: r for r in trace['requests']}
    mapping = {rid: [None] * len(r['output_token_ids']) for rid, r in requests.items()}
    for i, batch in enumerate(trace['batches']):
        for out in batch['outputs']:
            rid = out['request_id']
            start = out.get('token_start', out.get('position', 0))
            end = out.get('token_end', start + 1)
            for position in range(start, end):
                assert mapping[rid][position] is None, (rid, position, 'duplicate')
                assert requests[rid]['token_times_s'][position] == batch['time_s']
                mapping[rid][position] = i
    assert all(i is not None for values in mapping.values() for i in values)
    return mapping


def analyze_run(index, result, folder, trace, mapping, semantics, final_states):
    assert read(folder / 'result.json') == result, (folder, 'aggregate mismatch')
    server, client, producer = [read(folder / (name + '.json'))
                                for name in ('server', 'client', 'producer')]
    arm = ORDER[index]
    assert result['arm'] == server['arm'] == arm
    assert result['materialization'] == server['materialization']
    assert result['output_processor'] == server['output_processor']
    assert result['semantics'] == semantics
    assert not result['profile']
    assert result['all_completed']
    assert server['affinity'] == [2] and producer['affinity'] == [6]
    assert server['native_default_chunk_size'] == 1
    mat = server['materialization']
    assert mat['arm'] == arm and mat['max_wait_ms'] == 75
    assert mat['pending_bytes_at_end'] == mat['active_gates_at_end'] == 0
    assert mat['final_states'] == final_states
    requests = {r['request_id']: r for r in trace['requests']}
    assert set(requests) == set(client) == set(server['ledger']) == set(final_states)
    assert result['requests'] == len(requests)
    total_tokens = sum(len(x) for x in mapping.values())
    assert result['tokens'] == mat['stats']['tokens_materialized'] == total_tokens
    heavy_tokens = sum(len(mapping[rid]) for rid, r in requests.items() if r['heavy'])
    assert mat['stats']['sample_positions_materialized'] == heavy_tokens
    assert mat['stats']['sample_positions_deferred'] == (heavy_tokens if arm == 'lazy_static' else 0)
    epoch, events = producer['epoch_ns'], producer['events']
    assert server['epoch_ns'] == epoch
    assert [e[0] for e in events] == list(range(len(trace['batches'])))
    received = dict(server['source_received'])
    assert [e[0] for e in server['source_received']] == list(range(len(events)))
    for event, batch in zip(events, trace['batches']):
        assert event[1] == epoch + int(batch['time_s'] * 1e9)
        assert event[1] <= event[2] <= event[3]
        assert event[2] <= received[event[0]]
    assert [e[2] for e in events] == sorted(e[2] for e in events)
    pools = {kind: {'delivery': [], 'planned': [], 'gaps': [], 'completion': [],
                    'drain': [], 'source_receive': [], 'receive_collect': [],
                    'collect_visible': [], 'counts': Counter()}
             for kind in ('light', 'heavy')}
    per_request = {}
    for rid, request in requests.items():
        item, ledger = client[rid], server['ledger'][rid]
        kind = 'heavy' if request['heavy'] else 'light'
        pool = pools[kind]
        assert len(ledger) == len(item['visible_ns']) == item['chunks']
        assert item['visible_ns'] == sorted(item['visible_ns'])
        assert item['complete_ns'] >= max(item['visible_ns'])
        assert item['finish_reason'] == 'length'
        assert item['usage']['completion_tokens'] == len(mapping[rid])
        assert {k: item[k] for k in SEMA} == semantics[rid]
        assert item['logprobs_count'] == (len(mapping[rid]) if request['heavy'] else 0)
        final = final_states[rid]
        assert final['tokens'] == len(mapping[rid])
        assert final['logprob_positions'] == item['logprobs_count']
        assert final['finish_reason'] == 'length' and final['stop_reason'] is None
        assert (final['cumulative_logprob'] is not None) == bool(request['heavy'])
        delivery, cursor = [], 0
        for entry, visible in zip(ledger, item['visible_ns']):
            count, consumed = entry['token_count'], entry['consumed_ns']
            assert count > 0 and cursor + count <= len(mapping[rid])
            pool['counts'][count] += 1
            for batch_index in mapping[rid][cursor:cursor + count]:
                planned, emitted = events[batch_index][1:3]
                recv = received[batch_index]
                assert emitted <= recv <= consumed <= visible
                value = (visible - emitted) / 1e6
                delivery.append(value)
                pool['delivery'].append(value)
                pool['planned'].append((visible - planned) / 1e6)
                pool['source_receive'].append((recv - emitted) / 1e6)
                pool['receive_collect'].append((consumed - recv) / 1e6)
                pool['collect_visible'].append((visible - consumed) / 1e6)
            cursor += count
        assert cursor == len(mapping[rid])
        gaps = [(b - a) / 1e6 for a, b in zip(item['visible_ns'], item['visible_ns'][1:])]
        completion = (item['complete_ns'] - epoch) / 1e9 - request['arrival_s']
        drain = (item['complete_ns'] - events[mapping[rid][-1]][2]) / 1e6
        pool['gaps'].extend(gaps)
        pool['completion'].append(completion)
        pool['drain'].append(drain)
        group = 'background' if '-background-' in rid else 'long' if '-long-' in rid else 'other'
        per_request[rid] = {'kind': kind, 'group': group, 'tokens': len(mapping[rid]),
            'chunks': item['chunks'], 'bytes': item['bytes'],
            'delivery_p99_ms': q(delivery, .99), 'delivery_p50_ms': q(delivery, .5),
            'gap_p99_ms': q(gaps, .99), 'completion_s': completion,
            'drain_ms': drain, 'first_delivery_ms': delivery[0]}
    assert sum(item['bytes'] for item in client.values()) == result['bytes']
    assert sum(item['chunks'] for item in client.values()) == result['chunks']
    # Collector merging is allowed, so materialization count may exceed SSE count.
    assert mat['stats']['request_outputs_materialized'] >= result['chunks']
    elapsed = (max(item['complete_ns'] for item in client.values()) - epoch) / 1e9
    for key, value in [('elapsed_s', elapsed), ('tokens_s', total_tokens / elapsed),
                       ('req_s', len(requests) / elapsed),
                       ('server_cpu_s', server['cpu_seconds']),
                       ('server_peak_rss_mib', server['peak_rss_kib'] / 1024)]:
        same(value, result[key], (folder, key))
    for kind, pool in pools.items():
        for percentile in (.5, .95, .99):
            same(q(pool['delivery'], percentile), result[f'{kind}_delivery_p{int(percentile*100)}_ms'],
                 (folder, kind, percentile))
        for key, value in [('planned_delivery_p99_ms', q(pool['planned'], .99)),
                           ('gap_p99_ms', q(pool['gaps'], .99)),
                           ('completion_p99_s', q(pool['completion'], .99)),
                           ('drain_max_ms', max(pool['drain']))]:
            same(value, result[f'{kind}_{key}'], (folder, kind, key))
    late = [(e[2] - e[1]) / 1e6 for e in events]
    intervals = [1000 * (b['time_s'] - a['time_s'])
                 for a, b in zip(trace['batches'], trace['batches'][1:])]
    interval_error = [abs((b[2] - a[2] - (b[1] - a[1])) / 1e6)
                      for a, b in zip(events, events[1:])]
    same(q(late, .99), result['producer_lateness_p99_ms'], (folder, 'source late'))
    pacing = {'lateness_ms': describe(late), 'absolute_interbatch_error_ms': describe(interval_error),
              'predeclared_warning_threshold_ms': q(intervals, .5),
              'warning': q(late, .99) > q(intervals, .5)}
    metrics = {k: v for k, v in result.items() if k not in ('semantics', 'materialization')}
    metrics['total_frontend_cpu_s'] = result['server_cpu_s'] + result['client_cpu_s']
    categories = {}
    for kind, pool in pools.items():
        members = {rid: r for rid, r in per_request.items() if r['kind'] == kind}
        worst = max(members, key=lambda rid: members[rid]['delivery_p99_ms'])
        categories[kind] = {'request_delivery_p99_ms': describe([r['delivery_p99_ms'] for r in members.values()]),
                            'worst_request': worst, 'worst_request_metrics': members[worst],
                            'chunk_token_histogram': dict(sorted(pool['counts'].items())),
                            'stage_ms': {key: describe(pool[key]) for key in
                                         ('source_receive', 'receive_collect', 'collect_visible')}}
    return {'index': index, 'repetition': 0 if index < len(ARMS) else 1,
            'arm': arm, 'run': folder.name, 'metrics': metrics, 'pacing': pacing,
            'materialization': mat, 'categories': categories, 'requests': per_request,
            'versions': server['versions'],
            'input_sha256': {p.name: sha(p) for p in [folder / f'{n}.json' for n in
                            ('result', 'server', 'client', 'producer')]},
            'checks': {'flattened_semantic_hashes_equal': True, 'final_generation_states_equal': True,
                       'complete_token_ledger': True, 'finished_and_usage_correct': True,
                       'no_pending_state_at_end': True, 'sample_positions_preserved': True,
                       'source_plan_and_batch_order': True, 'causal_timestamps': True,
                       'recomputed_delivery_and_throughput': True, 'fixed_affinity_and_quota': True}}


def compare(candidate, baseline):
    keys = ('total_frontend_cpu_s', 'server_cpu_s', 'client_cpu_s', 'tokens_s', 'bytes', 'chunks',
            'light_delivery_p99_ms', 'heavy_delivery_p99_ms', 'light_gap_p99_ms', 'heavy_gap_p99_ms',
            'light_completion_p99_s', 'heavy_completion_p99_s',
            'server_peak_rss_mib', 'client_peak_rss_mib')
    metrics = {}
    for key in keys:
        a, b = candidate['metrics'][key], baseline['metrics'][key]
        metrics[key] = {'baseline': b, 'candidate': a, 'delta': a - b,
                        'delta_pct': 100 * (a / b - 1) if b else None}
    requests = {}
    for rid, a in candidate['requests'].items():
        b = baseline['requests'][rid]
        requests[rid] = {key: a[key] for key in ('kind', 'group', 'tokens')}
        for key in ('delivery_p99_ms', 'gap_p99_ms', 'completion_s', 'drain_ms', 'first_delivery_ms'):
            requests[rid][key] = {'baseline': b[key], 'candidate': a[key],
                                 'delta': a[key] - b[key] if a[key] is not None and b[key] is not None else None}
    grouped = []
    for kind in ('light', 'heavy'):
        for group in ('all', 'background', 'long', 'other'):
            members = {rid: r for rid, r in requests.items()
                       if r['kind'] == kind and (group == 'all' or r['group'] == group)}
            if not members:
                continue
            worst = max(members, key=lambda rid: members[rid]['delivery_p99_ms']['delta'])
            row = {'kind': kind, 'group': group, 'requests': len(members),
                   'tokens': sum(r['tokens'] for r in members.values()), 'worst_request': worst,
                   'worst_request_metrics': members[worst]}
            for key in ('delivery_p99_ms', 'gap_p99_ms', 'completion_s', 'drain_ms'):
                values = [r[key]['delta'] for r in members.values() if r[key]['delta'] is not None]
                row[key] = describe(values)
                row[key]['min'] = min(values) if values else None
            delta = [r['delivery_p99_ms']['delta'] for r in members.values()]
            row['delivery_signs'] = {'lower': sum(x < 0 for x in delta), 'higher': sum(x > 0 for x in delta),
                                     'equal': sum(x == 0 for x in delta)}
            row['baseline_request_p99_median_ms'] = q([r['delivery_p99_ms']['baseline'] for r in members.values()], .5)
            row['candidate_request_p99_median_ms'] = q([r['delivery_p99_ms']['candidate'] for r in members.values()], .5)
            grouped.append(row)
    return {'candidate_run': candidate['run'], 'baseline_run': baseline['run'],
            'candidate': candidate['arm'], 'baseline': baseline['arm'],
            'repetition': candidate['repetition'],
            'source_pacing_warning': candidate['pacing']['warning'] or baseline['pacing']['warning'],
            'metrics': metrics, 'request_groups': grouped, 'requests': requests}


def analyze_load(root, name):
    folder = root / name
    tracepath = root.parent / 'inputs' / (name + '.json')
    assert sha(tracepath) == SHAS[name], (name, 'frozen input SHA mismatch')
    trace, results, command = read(tracepath), read(folder / 'results.json'), read(folder / 'command.json')
    assert [r['arm'] for r in results] == ORDER
    assert command['arms'].split(',') == ORDER
    assert [command[key] for key in ('server_core', 'client_core', 'producer_core')] == [2, 4, 6]
    assert not command['profile']
    mapping = mapping_for(trace)
    semantics = results[0]['semantics']
    final_states = results[0]['materialization']['final_states']
    historical = root.parent / f'formal_{name}' / '01_fixed1' / 'result.json'
    historical_check = '未找到既有固定1结果；已检查新组全部方案间一致'
    if historical.is_file():
        assert read(historical)['semantics'] == semantics, (name, 'historical semantics mismatch')
        historical_check = {'path': str(historical), 'sha256': sha(historical), 'equal': True}
    runs = [analyze_run(i, result, folder / f'{i:02d}_{ORDER[i]}', trace, mapping, semantics, final_states)
            for i, result in enumerate(results)]
    assert all(r['versions'] == runs[0]['versions'] for r in runs)
    comparisons = []
    for rep in (0, 1):
        current = {r['arm']: r for r in runs if r['repetition'] == rep}
        for arm in ARMS[1:]:
            comparisons.append(compare(current[arm], current['native1']))
        comparisons.append(compare(current['lazy_static'], current['class_static']))
    return {'status': '已完成冻结14次运行并重新计算；探索证据',
            'trace_sha256': sha(tracepath), 'results_sha256': sha(folder / 'results.json'),
            'command_sha256': sha(folder / 'command.json'), 'historical_semantics_check': historical_check,
            'runs': runs, 'comparisons': comparisons}


def f(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def report(summary):
    out = ['# 输出物化：完整前端与逐请求取舍', '',
           '所有结果为冻结CPU开放环回放的探索证据；每方案每负载两次独立进程运行。'
           '不设统一成功百分比、不计算token/请求伪独立置信区间、不挑最好一次。'
           '完整排空吞吐受固定引擎产出限制，不是最大承载能力或GPU完整服务收益。', '',
           'N1/N2/N4=原生stream_interval 1/2/4；T=4 token或75ms；B=1024原始payload字节或75ms；'
           'S=light立即、heavy用T；L=与S同释放规则、延迟heavy logprob容器物化。'
           'T/B/S/L首token和完成立即。75ms是研究等待预算，不是应用SLO或客户端交付保证。', '',
           'CPU包含测量窗口内完整server/client工作与排空，分别报告并将CPU秒相加表示总资源工作；'
           '不把可重叠墙钟相加。OP子阶段不含timer回调物化，不能代替server总CPU。'
           'client峰值RSS是同一driver进程累积水位，不能视为每臂独立峰值；server为独立进程。', '']
    out += ['## 主结果表', '',
            '斜线依次为正序/反序重复，均保留。CPU为server+client总CPU秒；'
            '吞吐为所有响应完整排空后的request/s，取该方案两次范围。'
            '各类gap、逐请求副作用、分别的CPU与内存见下方原始重复表。', '']
    main_rows = []
    for name, load in summary['loads'].items():
        if 'runs' not in load:
            continue
        for arm in ARMS:
            pair = sorted([r for r in load['runs'] if r['arm'] == arm], key=lambda r: r['repetition'])
            def paired(key, digits=2):
                return '/'.join(f(r['metrics'][key], digits) for r in pair)
            rates = [r['metrics']['req_s'] for r in pair]
            main_rows.append([name, arm, paired('light_delivery_p99_ms'),
                              paired('heavy_delivery_p99_ms'), paired('total_frontend_cpu_s', 3),
                              paired('server_cpu_pct', 1), paired('server_peak_rss_mib', 1),
                              f'{min(rates):.4f}–{max(rates):.4f}'])
    out += table(['负载', '方案', 'light P99 ms', 'heavy P99 ms', '总CPU s',
                  'server单核CPU %', 'server峰值MiB', '完整排空request/s'], main_rows) + ['']
    for name, load in summary['loads'].items():
        out += [f'## {name}', '', load['status'], '']
        if 'runs' not in load:
            continue
        runs = load['runs']
        out += ['### 全部运行：成本和产出', '']
        rows = []
        for run in runs:
            m = run['metrics']
            rows.append([run['run'], f(m['server_cpu_s']), f(m['client_cpu_s']),
                         f(m['total_frontend_cpu_s']), f(m['server_peak_rss_mib'], 1),
                         f(m['client_peak_rss_mib'], 1), f(m['tokens_s'], 2), m['chunks'], m['bytes']])
        out += table(['运行', 'server CPU s', 'client CPU s', '合计CPU s', 'server峰值MiB',
                      'client水位MiB', '排空token/s', 'SSE chunks', 'bytes'], rows) + ['']
        out += ['### 全部运行：交付及逐请求尾部', '']
        rows = []
        for run in runs:
            m = run['metrics']
            rows.append([run['run'], f(m['light_delivery_p99_ms']), f(m['heavy_delivery_p99_ms']),
                         f(m['light_gap_p99_ms']), f(m['heavy_gap_p99_ms']),
                         f(run['categories']['light']['request_delivery_p99_ms']['max']),
                         f(run['categories']['heavy']['request_delivery_p99_ms']['max']),
                         f(m['light_completion_p99_s']), f(m['heavy_completion_p99_s'])])
        out += table(['运行', 'light P99 ms', 'heavy P99 ms', 'light gap P99 ms', 'heavy gap P99 ms',
                      '最坏light请求P99 ms', '最坏heavy请求P99 ms', 'light完成P99 s', 'heavy完成P99 s'], rows) + ['']
        out += ['### 主要同规则比较：L−S', '', '负数表示候选值降低；两次独立重复分别呈现。']
        ls = [c for c in load['comparisons'] if c['baseline'] == 'class_static']
        rows = []
        for comp in ls:
            m = comp['metrics']
            rows.append([comp['repetition'], f(m['total_frontend_cpu_s']['delta']),
                         f(m['total_frontend_cpu_s']['delta_pct'], 2),
                         f(m['light_delivery_p99_ms']['delta']), f(m['heavy_delivery_p99_ms']['delta']),
                         f(m['heavy_gap_p99_ms']['delta']), f(m['tokens_s']['delta_pct'], 3),
                         '时序失配，保留' if comp['source_pacing_warning'] else '通过源时序筛查'])
        out += [''] + table(['重复', '总CPU Δs', '总CPU Δ%', 'light P99 Δms', 'heavy P99 Δms',
                             'heavy gap Δms', '排空吞吐 Δ%', '源时序'], rows) + ['']
        out += ['### 相对原生N1的服务取舍', '',
                '这是全部预定点的探索性比较，不能从不同方案或不同重复拼接出一个未运行策略。', '']
        rows = []
        for comp in load['comparisons']:
            if comp['baseline'] != 'native1':
                continue
            m = comp['metrics']
            rows.append([comp['repetition'], comp['candidate'], f(m['total_frontend_cpu_s']['delta']),
                         f(m['light_delivery_p99_ms']['delta']), f(m['heavy_delivery_p99_ms']['delta']),
                         f(m['light_gap_p99_ms']['delta']), f(m['heavy_gap_p99_ms']['delta'])])
        out += table(['重复', '方案−N1', '总CPU Δs', 'light P99 Δms', 'heavy P99 Δms',
                      'light gap Δms', 'heavy gap Δms'], rows) + ['']
        out += ['### 逐请求代价：L−S与S−N1', '',
                '每请求先算自身交付P99，再按请求等权汇总配对差；符号计数不是显著性证据。'
                'background/long使用原输入身份，没有搜索输出长度阈值。完整每请求P99、gap、完成与最坏请求信息保存在summary.json。', '']
        rows = []
        selected = [c for c in load['comparisons'] if c['baseline'] == 'class_static' or c['candidate'] == 'class_static']
        for comp in selected:
            for group in comp['request_groups']:
                signs = group['delivery_signs']
                rows.append([f"{comp['candidate']}−{comp['baseline']}/{comp['repetition']}",
                             group['kind'] + '/' + group['group'], group['requests'],
                             f"{signs['lower']}/{signs['higher']}/{signs['equal']}",
                             f(group['delivery_p99_ms']['p50']), f(group['delivery_p99_ms']['min']),
                             f(group['delivery_p99_ms']['max']), f(group['gap_p99_ms']['p50']),
                             f(group['completion_s']['p50']), group['worst_request']])
        out += table(['比较/重复', '组', '请求数', 'P99降/升/同', 'P99 Δ中位ms', 'P99 Δ最小ms',
                      'P99 Δ最大ms', 'gap Δ中位ms', '完成 Δ中位s', '最坏Δ请求'], rows) + ['']
        out += ['### 物化工作与源时序', '',
                'sample positions仍全部完成；调用数减少不等于逐position工作消除。'
                'timer实际超期和数组复制计入代价。时序异常不删除、不补跑。', '']
        rows = []
        for run in runs:
            s, p = run['materialization']['stats'], run['pacing']
            rows.append([run['run'], s['sample_materialization_calls'], s['sample_positions_materialized'],
                         s['raw_merge_calls'], s['raw_merge_copy_bytes'], s['request_outputs_materialized'],
                         f(s['timer_materialization_cpu_ns'] / 1e9), f(s['max_deadline_overrun_ms']),
                         f(p['lateness_ms']['p99']), f(p['lateness_ms']['max']),
                         '失配保留' if p['warning'] else '正常'])
        out += table(['运行', 'sample调用', 'sample位置', 'raw合并', 'raw复制B', '输出对象次数',
                      'timer CPU s', '期限超期max ms', '源迟到P99 ms', '源迟到max ms', '状态'], rows) + ['']
        threshold = runs[0]['pacing']['predeclared_warning_threshold_ms']
        out += [f'本负载源迟到P99预定告警门槛：原批间隔中位数 {f(threshold)} ms。'
                '每运行迟到P50/P95/P99/max与相邻批间隔绝对误差均在summary.json。', '',
                '已核验完整token ledger、跨方案flatten文本/logprob摘要、最终生成状态、结束/usage、'
                'pending清空、保留全部logprob位置、原source计划及批序、时间因果顺序。'
                '摘要核验依赖driver在测量后计算的哈希，性能原件不存储全部SSE文本；'
                'STOP/ABORT/Unicode等扩展语义由独立semantic_check验证，不由本length回放替代。', '']
    out += ['## 解释边界', '',
            '本分析器不自动宣布继续或停止，不因局部CPU、一个最好运行或策略名称认定贡献。'
            'L/S释放规则相同，但timer执行、合并边界和内部轨迹可以不同；chunk数不强制相同。'
            '新增动态方法仍需当前可观测状态能解释静态方案系统性失配的独立证据。'
            '源失配或两次方向不一致时保留不确定性。所有结果与既有GPU-logprobs未验证、'
            '合成top20重复备选词、共享CPU主机及原始批内顺序缺失的限制同时解释。', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE,
                        help='materialization目录，内含high和/或medium完整14-run组')
    args = parser.parse_args()
    root = args.root.resolve()
    loads = {}
    for name in SHAS:
        if (root / name / 'results.json').is_file():
            loads[name] = analyze_load(root, name)
        else:
            loads[name] = {'status': '未分析：完整results.json尚未就绪（未运行或未导出，不是负结果）'}
    if not any('runs' in load for load in loads.values()):
        raise SystemExit('No complete frozen high/medium group found; no result artifacts written.')
    summary = {'schema': 1, 'evidence': 'exploratory_cpu_frontend_replay',
               'expected_order': ORDER, 'analysis_sha256': sha(Path(__file__)), 'loads': loads,
               'semantic_scope': 'native DELTA n=1 text/logprobs length replay; canonical flattened hashes',
               'no_token_or_request_independence_claim': True}
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (root / 'results.md').write_text(report(summary))
    print(json.dumps({'analyzed': [name for name, load in loads.items() if 'runs' in load],
                      'summary': str(root / 'summary.json'), 'report': str(root / 'results.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
