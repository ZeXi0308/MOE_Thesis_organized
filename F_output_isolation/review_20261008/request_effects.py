"""Post-hoc request-level effects of existing priority/FIFO pairs; no new runs.

Keep the frozen token-pooled primary metric unchanged. Requests share a run;
direction counts below are descriptive, not independent efficacy trials.
"""
from pathlib import Path
import json
import statistics

from analyze_endpoints import q, read, sha, token_batches, markdown_table, fmt, EXPECTED_SHA

ROOT = Path(__file__).resolve().parent
PAIRS = [('00_fixed1', '01_priority'), ('07_fixed1', '06_priority')]


def request_metrics(folder, trace, mapping):
    server, client, producer, result = [read(folder / (n + '.json'))
        for n in ('server', 'client', 'producer', 'result')]
    assert result['all_completed']
    output = {}
    pooled = {'light': [], 'heavy': []}
    for req in trace['requests']:
        rid = req['request_id']
        item = client[rid]
        values = []
        cursor = 0
        ledger = server['ledger'][rid]
        assert len(ledger) == len(item['visible_ns'])
        for entry, visible in zip(ledger, item['visible_ns']):
            for batch in mapping[rid][cursor:cursor + entry['token_count']]:
                emitted = producer['events'][batch][2]
                assert visible >= emitted
                values.append((visible - emitted) / 1e6)
            cursor += entry['token_count']
        assert cursor == len(mapping[rid]) == len(req['output_token_ids'])
        assert {k: item[k] for k in result['semantics'][rid]} == result['semantics'][rid]
        kind = 'heavy' if req['heavy'] else 'light'
        pooled[kind].extend(values)
        output[rid] = {'delivery_p99_ms': q(values, .99),
            'gap_p99_ms': q([(y-x)/1e6 for x,y in zip(item['visible_ns'],item['visible_ns'][1:])], .99),
            'completion_s': (item['complete_ns']-producer['epoch_ns'])/1e9-req['arrival_s'],
            'semantics': result['semantics'][rid]}
    for kind, values in pooled.items():
        assert abs(q(values, .99)-result[kind+'_delivery_p99_ms']) < 1e-8
    return output


def summarize(rows):
    deltas = [r['delta']['delivery_p99_ms'] for r in rows]
    return {'requests': len(rows), 'tokens': sum(r['tokens'] for r in rows),
        'lower': sum(x < 0 for x in deltas), 'higher': sum(x > 0 for x in deltas),
        'equal': sum(x == 0 for x in deltas),
        'baseline_median_request_p99_ms': statistics.median(r['baseline']['delivery_p99_ms'] for r in rows),
        'priority_median_request_p99_ms': statistics.median(r['priority']['delivery_p99_ms'] for r in rows),
        'delta_median_ms': statistics.median(deltas), 'delta_min_ms': min(deltas),
        'delta_max_ms': max(deltas),
        'worst_request': max(rows,key=lambda r:r['delta']['delivery_p99_ms'])['request_id']}


def order_audit(trace):
    """Describe saved input order and exact stable partition, not latency simulation."""
    requests = {r['request_id']:r for r in trace['requests']}
    indices = {rid:i for i,rid in enumerate(requests)}
    positions = {rid:[] for rid,r in requests.items() if r['heavy']}
    list_order_batches = 0
    arrival_inversion_batches = 0
    for batch in trace['batches']:
        ids = [o['request_id'] for o in batch['outputs']]
        ranks = [indices[rid] for rid in ids]
        list_order_batches += ranks == sorted(ranks)
        arrival_inversion_batches += any(requests[a]['arrival_s'] > requests[b]['arrival_s']
            for a,b in zip(ids,ids[1:]))
        partition = sorted(ids,key=lambda rid:bool(requests[rid]['heavy']))
        new_positions = {rid:i for i,rid in enumerate(partition)}
        for pos,rid in enumerate(ids):
            if requests[rid]['heavy']:
                passed = sum(not requests[x]['heavy'] for x in ids[pos+1:])
                assert new_positions[rid]-pos == passed
                positions[rid].append((pos,new_positions[rid],passed))
    per_request = []
    cohorts = []
    for cohort in ('background','long'):
        subset = [r for r in requests.values() if f'-{cohort}-' in r['request_id']]
        assert subset
        cohort_heavy = [r for r in subset if r['heavy']]
        passed = [row[2] for r in cohort_heavy for row in positions[r['request_id']]]
        cohorts.append({'cohort':cohort,'requests':len(subset),
            'output_lengths':sorted(set(len(r['output_token_ids']) for r in subset)),
            'prompt_length_range':[min(len(r['prompt_token_ids']) for r in subset),max(len(r['prompt_token_ids']) for r in subset)],
            'arrival_values_s':sorted(set(r['arrival_s'] for r in subset)),
            'heavy_outputs':len(passed),'heavy_outputs_crossed_by_light':sum(v>0 for v in passed),
            'median_light_passes':statistics.median(passed)})
        for req in cohort_heavy:
            rows = positions[req['request_id']]
            per_request.append({'request_id':req['request_id'],'cohort':cohort,
                'arrival_s':req['arrival_s'],'tokens':len(req['output_token_ids']),
                'median_A_position':statistics.median(row[0] for row in rows),
                'median_P_position':statistics.median(row[1] for row in rows),
                'median_light_passes':statistics.median(row[2] for row in rows),
                'max_light_passes':max(row[2] for row in rows)})
    return {'scope':'输入顺序及P的精确稳定分区算数；非延迟模拟，非GPU原始顺序',
        'batches':len(trace['batches']),'batches_in_global_request_list_order':list_order_batches,
        'batches_with_adjacent_arrival_inversion':arrival_inversion_batches,
        'cohorts':cohorts,'heavy_request_positions':per_request}


def main():
    trace_path = ROOT.parent / 'inputs/high.json'
    assert sha(trace_path) == EXPECTED_SHA
    trace = read(trace_path)
    mapping = token_batches(trace)
    pairs = []
    source_hashes = {'inputs/high.json': sha(trace_path)}
    for rep, (baseline, priority) in enumerate(PAIRS):
        metrics = []
        for name in (baseline, priority):
            folder = ROOT / 'endpoint_high' / name
            metrics.append(request_metrics(folder, trace, mapping))
            for stem in ('server', 'client', 'producer', 'result'):
                file = folder / (stem+'.json')
                source_hashes[str(file.relative_to(ROOT))] = sha(file)
        rows = []
        for req in trace['requests']:
            rid = req['request_id']
            a, p = metrics[0][rid], metrics[1][rid]
            assert a['semantics'] == p['semantics']
            fields = ('delivery_p99_ms', 'gap_p99_ms', 'completion_s')
            rows.append({'request_id': rid, 'class': 'heavy' if req['heavy'] else 'light',
                'tokens': len(req['output_token_ids']), 'arrival_s': req['arrival_s'],
                'baseline': {k:a[k] for k in fields}, 'priority': {k:p[k] for k in fields},
                'delta': {k:p[k]-a[k] for k in fields}})
        groups = []
        for kind in ('light', 'heavy'):
            selected = [r for r in rows if r['class']==kind]
            groups.append({'class':kind, 'length':'all', **summarize(selected)})
            for n in sorted(set(r['tokens'] for r in selected)):
                groups.append({'class':kind, 'length':n,
                    **summarize([r for r in selected if r['tokens']==n])})
        pairs.append({'repetition':rep, 'baseline':baseline, 'priority':priority,
            'groups':groups, 'requests':rows})
    cross = {}
    for kind in ('light','heavy'):
        by_run = [{r['request_id']:r for r in p['requests'] if r['class']==kind} for p in pairs]
        ids = sorted(by_run[0])
        cross[kind] = {'requests':len(ids),
            'lower_both':sum(all(b[r]['delta']['delivery_p99_ms']<0 for b in by_run) for r in ids),
            'higher_both':sum(all(b[r]['delta']['delivery_p99_ms']>0 for b in by_run) for r in ids),
            'higher_both_ids':[r for r in ids if all(b[r]['delta']['delivery_p99_ms']>0 for b in by_run)]}
    out = {'status':'后验逐请求描述；原始主指标不变、无新增运行、无独立请求置信区间',
        'source_sha256':source_hashes, 'analyzer_sha256':sha(Path(__file__)),
        'pairs':pairs, 'cross_pair_direction':cross, 'saved_order_audit':order_audit(trace)}
    (ROOT/'request_effects.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 逐请求收益分布：复用现有P/A两对记录', '',
        '这是后验描述，补查token汇总是否掩盖请求子群损失。原主指标不变，不新增运行。每个请求先计算自身token交付P99；下表按请求等权汇总这些值，与合并全部token的P99不同。方向计数包含微小变化，不代表显著性；请求共享运行，不能当作独立重复。', '',
        '只按输入本来就有的精确输出长度列组，不搜索阈值或挑选请求。Δ=priority−FIFO；负值表示该请求的交付P99下降。中位数Δ是配对差的中位数，不是两列中位数之差。', '']
    lines += markdown_table(['重复/类别/输出长度','请求数 / token数','P99下降/上升/相同请求数','请求P99中位数 A→P ms','配对Δ中位数 ms','配对Δ最小/最大 ms'],[
        [f"{p['repetition']}/{g['class']}/{g['length']}",f"{g['requests']} / {g['tokens']}",
         f"{g['lower']}/{g['higher']}/{g['equal']}",f"{fmt(g['baseline_median_request_p99_ms'])}→{fmt(g['priority_median_request_p99_ms'])}",
         fmt(g['delta_median_ms']),f"{fmt(g['delta_min_ms'])}/{fmt(g['delta_max_ms'])}"]
        for p in pairs for g in p['groups']])
    lines += ['', '两对均同方向（仍不是新增独立重复）：', '']
    lines += markdown_table(['类别','请求数','两对均下降','两对均上升'],[
        [kind,c['requests'],c['lower_both'],c['higher_both']] for kind,c in cross.items()])
    order = out['saved_order_audit']
    lines += ['', '## 固定顺序与工作负载组的混杂', '',
        f"{order['batches_in_global_request_list_order']}/{order['batches']}批严格按全局requests列表排列，先background、后long；其中{order['batches_with_adjacent_arrival_inversion']}批有相邻到达时间逆序，因此这也不是严格的按到达先后服务。原GPU批内顺序已缺失。下面只计算已保存顺序及P的精确稳定分区，不模拟延迟或把该顺序外推真实引擎。", '']
    lines += markdown_table(['组','请求数','输出长度','prompt长度范围','重输出数/被light跨越数','跨越light数中位数'],[
        [c['cohort'],c['requests'],str(c['output_lengths']),str(c['prompt_length_range']),
         f"{c['heavy_outputs']}/{c['heavy_outputs_crossed_by_light']}",fmt(c['median_light_passes'])]
        for c in order['cohorts']])
    lines += ['', '八个重background请求的位置（从0开始，中位数按各自输出计算）：', '']
    lines += markdown_table(['请求','到达s','A位置中位数','P位置中位数','跨越light中位数/最大'],[
        [r['request_id'],r['arrival_s'],fmt(r['median_A_position']),fmt(r['median_P_position']),
         f"{fmt(r['median_light_passes'])}/{r['max_light_passes']}"]
        for r in order['heavy_request_positions'] if r['cohort']=='background'])
    lines += ['', '输出长度与工作负载组完全重合，prompt长度及到达计划也不同。因此应称“原固定顺序中靠前的重background子群退化”，不能称“短输出导致退化”。顺序算数与退化相容，但不能定量归因每毫秒差异。', '',
        '120个轻请求两对均受益，说明汇总改善并非只来自少数轻请求。与此同时，原重类总体P99的小增幅掩盖了background子群集中代价；P只能保留为有代价的对照，不能认定已解决公平隔离。', '',
        '每个请求的交付P99、gap P99、完成时间和配对差完整保存在request_effects.json；原始文件SHA一并保存。请求内P99仅由有限尾部样本决定，且token/请求相关；本分析不提供显著性证明。它也不能证明全部获益由logprobs成本差异造成，不能证明真实部署重要性、未来无饥饿或方法新颖性。']
    (ROOT/'request_effects.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cross_pair_direction':cross,'groups':[p['groups'] for p in pairs]},ensure_ascii=False))


if __name__ == '__main__':
    main()
