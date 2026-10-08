"""Bounded post-hoc localization of endpoint tails; no new measurements.

All eight runs and all data remain in the main result. Removal sensitivities
below are descriptive checks of startup concentration, never revised results.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from analyze_endpoints import ORDER, LABELS, STAGES, q, read, sha, token_batches, markdown_table, fmt

ROOT = Path(__file__).resolve().parent


def audit(index, arm, trace, mapping):
    run = ROOT / 'endpoint_high' / f'{index:02d}_{arm}'
    server, client, producer = [read(run / (name + '.json')) for name in ('server', 'client', 'producer')]
    received = dict(server['source_received'])
    events = producer['events']
    epoch = producer['epoch_ns']
    samples = {'light': [], 'heavy': []}
    per_batch = {i: [] for i in range(len(events))}
    for request in trace['requests']:
        rid = request['request_id']
        kind = 'heavy' if request['heavy'] else 'light'
        cursor = 0
        for item, visible in zip(server['ledger'][rid], client[rid]['visible_ns']):
            for batch in mapping[rid][cursor:cursor + item['token_count']]:
                emitted, consume = events[batch][2], item['consumed_ns']
                row = {'batch': batch, 'kind': kind, 'emitted_ns': emitted,
                       'consumed_ns': consume, 'visible_ns': visible,
                       'delivery_ms': (visible - emitted) / 1e6,
                       'emit_receive_ms': (received[batch] - emitted) / 1e6,
                       'receive_collector_ms': (consume - received[batch]) / 1e6,
                       'collector_client_ms': (visible - consume) / 1e6}
                samples[kind].append(row)
                per_batch[batch].append(row)
            cursor += item['token_count']
        assert cursor == len(mapping[rid])
    # A transparent cold-start exposure definition: outputs emitted before the
    # final visible token of batch 0. This is descriptive, not a steady-state cut.
    first_final_visible = max(row['visible_ns'] for row in per_batch[0])
    cold_batches = [i for i, event in enumerate(events) if event[2] <= first_final_visible]
    result = {'run': run.name, 'arm': arm, 'cold_exposure_definition':
              'batch emitted no later than the final visible token of batch 0',
              'cold_exposure_batches': cold_batches,
              'first_batch_final_visible_since_emit_ms': (first_final_visible - events[0][2]) / 1e6,
              'classes': {}}
    for kind, rows in samples.items():
        threshold = q([row['delivery_ms'] for row in rows], .99)
        tail = [row for row in rows if row['delivery_ms'] >= threshold]
        counts = Counter(row['batch'] for row in tail)
        top = counts.most_common()
        covers = {}
        for fraction in (.5, .9):
            total = 0
            for n, (_, count) in enumerate(top, 1):
                total += count
                if total >= fraction * len(tail):
                    covers[str(fraction)] = n
                    break
        details = []
        for batch, count in top[:10]:
            prior = per_batch.get(batch - 1, [])
            batch_rows = per_batch[batch]
            details.append({'batch': batch, 'tail_tokens': count,
                'emit_since_epoch_s': (events[batch][2] - epoch) / 1e9,
                'producer_lateness_ms': (events[batch][2] - events[batch][1]) / 1e6,
                'emit_receive_ms': (received[batch] - events[batch][2]) / 1e6,
                'receive_to_last_collector_ms': (max(row['consumed_ns'] for row in batch_rows) - received[batch]) / 1e6,
                'batch_post_max_ms': max(row['collector_client_ms'] for row in batch_rows),
                'previous_batch_receive_to_last_collector_ms': None if not prior else
                    (max(row['consumed_ns'] for row in prior) - received[batch - 1]) / 1e6,
                'previous_batch_last_collector_to_this_receive_ms': None if not prior else
                    (received[batch] - max(row['consumed_ns'] for row in prior)) / 1e6})
        dominant = Counter(max(STAGES, key=lambda key: row[key]) for row in tail)
        means = {key: statistics.mean(row[key] for row in tail) for key in (*STAGES, 'delivery_ms')}
        sorted_batches = sorted(counts)
        episodes = []
        for batch in sorted_batches:
            if not episodes or batch > episodes[-1]['end'] + 1:
                episodes.append({'start': batch, 'end': batch, 'tail_tokens': counts[batch]})
            else:
                episodes[-1]['end'] = batch
                episodes[-1]['tail_tokens'] += counts[batch]
        result['classes'][kind] = {
            'delivery_p99_ms': threshold, 'tokens': len(rows), 'tail_tokens': len(tail),
            'tail_batches': len(counts), 'batches_covering_tail_fraction': covers,
            'top5_batches_tail_fraction': sum(count for _, count in top[:5]) / len(tail),
            'batch0_tail_tokens': counts[0],
            'cold_exposure_tail_tokens': sum(counts[batch] for batch in cold_batches),
            'tail_dominant_stage_token_counts': dict(dominant),
            'tail_conditional_mean_ms': means,
            'tail_stage_mean_fraction': {key: means[key] / means['delivery_ms'] for key in STAGES},
            'sensitivity_only_p99_without_batch0_ms': q([row['delivery_ms'] for row in rows if row['batch'] != 0], .99),
            'sensitivity_only_p99_without_cold_exposure_ms': q([row['delivery_ms'] for row in rows if row['batch'] not in cold_batches], .99),
            'top10_tail_batches': details, 'tail_batch_episodes': episodes}
    return result


def main():
    trace_path = ROOT.parent / 'inputs/high.json'
    trace = read(trace_path)
    mapping = token_batches(trace)
    runs = [audit(i, arm, trace, mapping) for i, arm in enumerate(ORDER)]
    summary = {'status': '后验描述诊断；未删除主分析数据、未增加运行、不可单凭时间定位推断GC或OS原因',
               'trace_sha256': sha(trace_path), 'analyzer_sha256': sha(Path(__file__)), 'runs': runs}
    (ROOT / 'tail_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 端点尾延迟定位（后验、只读原始记录）', '',
        '本审查只定位已有8次运行中的等待阶段与批次集中度。未新跑、未改方法、未删主结果。每run仍是独立重复单位；下面token和batch计数都是描述。', '',
        '尾集定义为同run、同类delivery ≥ 该类P99，分段均值只在这个相同尾集上计算。source表示producer实际emit→服务端handler接收批号，并非producer迟到。', '']
    lines += markdown_table(['运行/臂', 'light P99 ms', '尾均值 source/server/post ms', '尾source/server/post占比%',
                              '尾批次数；50%/90%尾所需批次', '前5批占尾%', 'batch0尾数/总尾', '冷暴露尾数/总尾'], [
        [f"{r['run']}/{LABELS[r['arm']]}", fmt(d['delivery_p99_ms']),
         '/'.join(fmt(d['tail_conditional_mean_ms'][key]) for key in STAGES),
         '/'.join(fmt(100*d['tail_stage_mean_fraction'][key], 1) for key in STAGES),
         f"{d['tail_batches']}; {d['batches_covering_tail_fraction']['0.5']}/{d['batches_covering_tail_fraction']['0.9']}",
         fmt(100*d['top5_batches_tail_fraction'], 1), f"{d['batch0_tail_tokens']}/{d['tail_tokens']}",
         f"{d['cold_exposure_tail_tokens']}/{d['tail_tokens']}"] for r in runs for d in [r['classes']['light']]])
    lines += ['', '冷暴露采用明确的描述定义：batch 0最后一个token客户端可见之前已emit的批次。以下剔除敏感性仅回答是否由开头数据决定，**不替换正式P99或用于挑选数据**。', '']
    lines += markdown_table(['运行', '冷暴露batch编号', 'batch0最终可见等待 ms', '原P99 ms', '除batch0敏感性 ms', '除冷暴露敏感性 ms'], [
        [r['run'], ','.join(map(str,r['cold_exposure_batches'])), fmt(r['first_batch_final_visible_since_emit_ms']),
         fmt(d['delivery_p99_ms']), fmt(d['sensitivity_only_p99_without_batch0_ms']),
         fmt(d['sensitivity_only_p99_without_cold_exposure_ms'])] for r in runs for d in [r['classes']['light']]])
    for r in runs:
        if r['arm'] != 'cache_priority':
            continue
        lines += ['', f"## {r['run']}：light尾token最多的10个批次", '',
                  'previous列用于定位前一批内/前一collector后是否已有等待，不足以确认是哪段代码、GC、OS抢占或其他任务导致。最后一列可略小于零：handler可以先读取下一批，前一批的最后consumer随后才恢复；这是异步交错，不是负的服务耗时。', '']
        lines += markdown_table(['batch', '尾token', 'emit相对epoch s', 'producer迟到 ms', 'emit→receive ms',
                                  '本批receive→末collector ms', '本批最大post ms',
                                  '前批receive→末collector ms', '前批末collector→本批receive ms'], [
            [v['batch'], v['tail_tokens'], fmt(v['emit_since_epoch_s']), fmt(v['producer_lateness_ms']),
             fmt(v['emit_receive_ms']), fmt(v['receive_to_last_collector_ms']), fmt(v['batch_post_max_ms']),
             fmt(v['previous_batch_receive_to_last_collector_ms']),
             fmt(v['previous_batch_last_collector_to_this_receive_ms'])]
            for v in r['classes']['light']['top10_tail_batches']])
    lines += ['', '完整两类分解、尾占比及连续批次段保存在tail_audit.json。聚合wall−CPU差、计时尖峰和阶段归属均不是GC或OS原因的充分证据。缓存与priority的组合关联到尖峰，两个运行仍不能区分确定的策略相互作用、运行顺序漂移及其他主机噪声。',
              '', '对模型的含义：批内顺序可控制接收后等待的局部模型仍可成立；假设“减小单位解码成本与priority效果可相加，因而完整交付必定改善”的模型在本组不获支持。必须计入前一批排空、handler读取机会和post阶段才能解释完整尾；现有记录没有提供足以设计可靠新在线状态的因果证据。']
    (ROOT / 'tail_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({r['run']: r['classes']['light'] for r in runs if r['arm'] == 'cache_priority'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
