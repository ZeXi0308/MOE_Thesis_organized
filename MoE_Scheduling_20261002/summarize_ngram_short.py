"""Summarize complete AR/N1/N2 runs using the existing natural-output analyzers.

One command (use the local Python environment that has tokenizers installed):
  python summarize_ngram_short.py --results /path/to/results_ngram_short_r01

The default reruns both analyzers in a temporary symlink view, so source captures
and existing metrics.json/ngram_activity.json are never overwritten. For a
format/regression check on old N4 results, pass --design ngram
--existing-analysis --output /tmp/ngram-summary.json --report /tmp/ngram-summary.md.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from analyze_kv_swap import ROOT, comparison, load, ratio


DESIGNS = {
    'ngram_short': {
        'AR16': ('00_ar16', '05_ar16'),
        'N1': ('01_ngram1', '04_ngram1'),
        'N2': ('02_ngram2', '03_ngram2'),
    },
    'ngram': {
        'AR16': ('00_ar16', '03_ar16'),
        'N4': ('01_ngram4', '02_ngram4'),
    },
}


def read_analysis(args, groups):
    names = sorted(name for repeats in groups.values() for name in repeats)
    if args.existing_analysis:
        return load(args.results / 'metrics.json'), load(args.results / 'ngram_activity.json')
    with tempfile.TemporaryDirectory(prefix='ngram-summary-') as temporary:
        view = Path(temporary)
        for name in [*names, 'group_status.json']:
            source = args.results / name
            if not source.exists():
                raise ValueError(f'缺少完整组数据：{source}')
            (view / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
        common = ['--results', str(view), '--design', args.design]
        subprocess.run([
            sys.executable, str(ROOT / 'analyze_kv_group.py'), *common,
            '--inputs', str(args.inputs), '--metadata', str(args.metadata),
            '--output', str(view / 'metrics.json'),
        ], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([
            sys.executable, str(ROOT / 'analyze_ngram_activity.py'), *common,
            '--output', str(view / 'ngram_activity.json'),
        ], check=True, stdout=subprocess.DEVNULL)
        return load(view / 'metrics.json'), load(view / 'ngram_activity.json')


def flat_metrics(cell, activity):
    cost, stages = cell['kv_service_cost'], cell['stages']
    values = {key: cell[key] for key in (
        'episode_wall_s', 'output_tokens', 'output_tokens_per_s', 'accuracy',
        'correct_requests', 'correct_and_eos_requests', 'cap_truncated_requests',
        'served_output_tokens_per_s', 'drained_output_tokens_per_s',
        'weight_copy_bytes', 'group_count', 'completed_requests',
    )}
    values.update({key: cost.get(key) for key in ('served_s', 'drained_s', 'capture_end_s')})
    values.update({f'{kind}_{stat}': cell[kind][stat]
                   for kind in ('flow', 'ttft', 'itl', 'inter_chunk_gap', 'request_maxgap')
                   for stat in ('mean_s', 'p95_s', 'max_s')})
    values.update(engine_wall_s=stages['total_engine_wall_s'],
                  engine_calls=stages['engine_calls'], scheduler_steps=stages['scheduler_steps'],
                  preemptions=cell['preemption']['total_preemption_events'],
                  scheduled_draft_token_rows=activity['scheduled_draft_token_rows'],
                  proposal_request_steps=activity['proposal_request_steps'],
                  multi_chunk_requests=len(activity['multi_chunk_request_ids']),
                  expert_bytes_per_output_token=ratio(cell['weight_copy_bytes'], cell['output_tokens']))
    values.update({f'kv_{key}_bytes': (cell.get('completed_transfer_bytes') or {}).get(key)
                   for key in ('load', 'store')})
    return values


def statistics_by_metric(rows):
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        observed = all(value is not None for value in values)
        result[key] = dict(values=values, n=len(values),
                           mean=statistics.mean(values) if observed else None,
                           min=min(values) if observed else None,
                           max=max(values) if observed else None)
    return result


def build_summary(metrics, activity, groups, results, design):
    names = sorted(name for repeats in groups.values() for name in repeats)
    if metrics.get('design') != design or metrics.get('group_status') != 'COMPLETE':
        raise ValueError('metrics 的 design 或 COMPLETE 状态与本次汇总不符')
    cells = metrics['cells']
    if set(cells) != set(names) or set(activity['cells']) != set(names):
        raise ValueError('metrics/activity 的运行清单与指定设计不符')
    if not all(cells[name]['all_16_complete'] for name in names):
        raise ValueError('不能把未完成请求组汇总成完整请求比较')
    inventories = [{row['request_id'] for row in cells[name]['per_request']} for name in names]
    if any(inventory != inventories[0] for inventory in inventories):
        raise ValueError('各运行的请求清单不一致')
    # The original comparison requires source order; align by request ID first.
    order = [row['request_id'] for row in cells[names[0]]['per_request']]
    for name in names:
        rows = {row['request_id']: row for row in cells[name]['per_request']}
        if len(rows) != len(cells[name]['per_request']):
            raise ValueError(f'{name}: 重复 request_id')
        cells[name] = {**cells[name], 'per_request': [rows[rid] for rid in order]}
    flat = {name: flat_metrics(cells[name], activity['cells'][name]) for name in names}
    stats = {group: statistics_by_metric([flat[name] for name in repeats])
             for group, repeats in groups.items()}
    means = {group: {key: value['mean'] for key, value in values.items()}
             for group, values in stats.items()}

    def pair(candidate, baseline):
        value = comparison(candidate, baseline, cells)
        value['candidate'], value['baseline'] = candidate, baseline
        value['metric_ratios'] = {key: ratio(flat[candidate][key], flat[baseline][key])
                                  for key in flat[candidate]}
        value['metric_differences'] = {
            key: flat[candidate][key] - flat[baseline][key]
            if flat[candidate][key] is not None and flat[baseline][key] is not None else None
            for key in flat[candidate]
        }
        value['identical_full_texts'] = sum(row['identical_text'] for row in value['per_request'])
        value['changed_sequence_request_ids'] = [row['request_id'] for row in value['per_request']
                                                  if not row['identical_token_ids']]
        value['changed_answer_request_ids'] = [row['request_id'] for row in value['per_request']
                                                if row['left_answer'] != row['right_answer']]
        return value

    comparisons = []
    for baseline, candidate in combinations(groups, 2):
        comparisons.append(dict(
            candidate=candidate, baseline=baseline,
            ratios_of_repeat_means={key: ratio(means[candidate][key], means[baseline][key])
                                    for key in means[candidate]},
            accuracy_difference=means[candidate]['accuracy'] - means[baseline]['accuracy'],
            pairs=[pair(groups[candidate][index], groups[baseline][index]) for index in (0, 1)],
        ))
    stage_kinds = sorted({kind for cell in cells.values() for kind in cell['stages']['by_kind']})
    stage_fields = ('engine_calls', 'scheduler_steps', 'prefill_tokens', 'decode_tokens',
                    'total_engine_wall_s', 'weight_copy_bytes', 'groups')
    stage_means = {
        group: {kind: {key: statistics.mean(
            cells[name]['stages']['by_kind'].get(kind, {}).get(key, 0) for name in repeats
        ) for key in stage_fields} for kind in stage_kinds}
        for group, repeats in groups.items()
    }
    output_fields = (
        'request_id', 'example_index', 'gold', 'answer', 'score', 'correct_and_eos',
        'finish_reason', 'stop_reason', 'cap_truncated', 'output_tokens',
        'output_text', 'output_token_ids', 'flow_s', 'ttft_s', 'maxgap_s',
        'token_level_itl_resolved',
    )
    return dict(
        evidence_type='ACTUAL_NATURAL_EOS_FIXED_NGRAM_RUN_SUMMARY', design=design,
        results_directory=str(results.resolve()), order=names, groups=groups,
        workload_sha256=metrics.get('workload_sha256'), tokenizer_sha256=metrics.get('tokenizer_sha256'),
        resource_checks_pass=metrics.get('all_resource_checks_pass'),
        activity_checks_pass=activity.get('all_checks_pass'),
        cell_metrics=flat, group_statistics=stats, repeat_means=means,
        aggregate_drained_rates={group: ratio(
            sum(flat[name]['output_tokens'] for name in repeats),
            sum(flat[name]['drained_s'] for name in repeats)
            if all(flat[name]['drained_s'] is not None for name in repeats) else None,
        ) for group, repeats in groups.items()},
        comparisons=comparisons,
        within_group_repeats={group: pair(repeats[1], repeats[0]) for group, repeats in groups.items()},
        expert_stage_means=stage_means,
        cell_output_details={name: [{key: row.get(key) for key in output_fields}
                                    for row in cells[name]['per_request']] for name in names},
        cell_finish_counts={name: cells[name]['finish_counts'] for name in names},
        cell_itl_resolved={name: cells[name]['token_level_itl_resolved'] for name in names},
        activity={name: {key: activity['cells'][name][key] for key in (
            'configured_speculative_tokens', 'draft_length_distribution',
            'positive_receipt_chunk_distribution', 'multi_chunk_request_ids',
            'output_tokens_beyond_one_per_positive_receipt', 'exact_acceptance_rate',
        )} for name in names},
        notes=[
            '重复单位为独立运行；配对是第 1 次对第 1 次、第 2 次对第 2 次；不是独立输入重复。',
            '表中均值为运行级统计量的算术均值，含各运行分位数的均值；不是将 token 当独立样本。',
            '真实输出率为实际返回 token / drained 秒；初始化及 warmup 不计入，执行与末尾 drain 不作减项。',
            '输出长度、文本或 token ID 改变时，时间比只表示实际 episode 比较，不能称等工作量加速或输出等价。',
            '正确率复用历史 last-number 字符串匹配；相同准确率不代表同样答案或完整输出。',
            'host chunk gap 排除 TTFT；多 token 同次返回时，无法恢复 chunk 内 token ITL，相关 ITL 为 null。',
            '调度 decode 位置包含验证候选；receipt 多出的 token 不等于 accepted drafts；exact acceptance 不可推断。',
            '专家 copy 为实际记录字节，不能直接转成暴露传输时间；engine 阶段时间包含等待且不另加 copy 时间。',
        ],
    )


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def change(value):
    return '—' if value is None else f'{100 * (value - 1):+.2f}%'


def render_report(summary):
    lines = ['# 固定短草稿完整请求汇总', '',
             f"设计：`{summary['design']}`；结果：`{summary['results_directory']}`。"
             '所有运行均完整保留。下面的变化是实际完整 episode 比较，输出工作量可能不同。', '',
             f"既有资源核对：{summary['resource_checks_pass']}；活动核对：{summary['activity_checks_pass']}。", '',
             '| 运行/两重复均值 | served / drained 秒 | 实际 token/s | mean flow 秒 | TTFT 均值秒 | chunk gap p95 / max 秒 | 输出 token | 正确/请求 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows = list(summary['cell_metrics'].items()) + [(f'{group} 均值', values)
                                                     for group, values in summary['repeat_means'].items()]
    for name, row in rows:
        lines.append(f"| {name} | {number(row['served_s'])} / {number(row['drained_s'])} | "
                     f"{number(row['drained_output_tokens_per_s'])} | {number(row['flow_mean_s'])} | "
                     f"{number(row['ttft_mean_s'])} | {number(row['inter_chunk_gap_p95_s'])} / "
                     f"{number(row['inter_chunk_gap_max_s'])} | {number(row['output_tokens'], 1)} | "
                     f"{number(row['correct_requests'], 1)}/{number(row['completed_requests'], 0)} |")
    lines.extend(['', '| 候选/基线 | drained | 实际 token/s | mean flow | 最大 host gap | 输出长度 | 专家 copy |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    keys = ('drained_s', 'drained_output_tokens_per_s', 'flow_mean_s', 'inter_chunk_gap_max_s',
            'output_tokens', 'weight_copy_bytes')
    for item in summary['comparisons']:
        rows = [(f"{item['candidate']}/{item['baseline']} 均值比", item['ratios_of_repeat_means'])]
        rows += [(f"{pair['candidate']}/{pair['baseline']}", pair['metric_ratios']) for pair in item['pairs']]
        lines.extend('| ' + name + ' | ' + ' | '.join(change(ratios[key]) for key in keys) + ' |'
                     for name, ratios in rows)
    lines.extend(['', '配对逐次比较完整 token 序列、文本、末尾答案和正确性；全文与 token IDs 保存在 JSON。', '',
                  '| 配对 | 完整 token 序列相同 | 完整文本相同 | 答案相同 | 候选独对 / 基线独对 |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for item in summary['comparisons']:
        for pair in item['pairs']:
            total = len(pair['per_request'])
            lines.append(f"| {pair['candidate']}/{pair['baseline']} | {pair['identical_full_sequences']}/{total} | "
                         f"{pair['identical_full_texts']}/{total} | {pair['same_answers']}/{total} | "
                         f"{pair['left_only_correct']} / {pair['right_only_correct']} |")
    lines.append('')
    for group, pair in summary['within_group_repeats'].items():
        stats = summary['group_statistics'][group]
        lines.append(f"- {group} 两重复：完整序列相同 {pair['identical_full_sequences']}/{len(pair['per_request'])}；"
                     f"drained 范围 {number(stats['drained_s']['min'])}–{number(stats['drained_s']['max'])} 秒；"
                     f"实际输出率范围 {number(stats['drained_output_tokens_per_s']['min'])}–"
                     f"{number(stats['drained_output_tokens_per_s']['max'])} token/s。")
    lines.extend(['', 'ITL 与 host chunk 分开解释：', ''])
    for name, resolved in summary['cell_itl_resolved'].items():
        values = summary['cell_metrics'][name]
        status = (f"可解析，mean/p95={number(values['itl_mean_s'])}/{number(values['itl_p95_s'])} 秒"
                  if resolved else '不可解析（null），不把同一 chunk 的重复时间当零 ITL')
        lines.append(f"- {name}：token ITL {status}；多 token chunk 请求 {values['multi_chunk_requests']} 个。")
    lines.extend(['', '| 臂/阶段均值 | steps | prefill / decode 位置 | engine wall 秒 | 专家 copy GB | groups |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for group, stages in summary['expert_stage_means'].items():
        for kind, values in stages.items():
            lines.append(f"| {group}/{kind} | {number(values['scheduler_steps'], 1)} | "
                         f"{number(values['prefill_tokens'], 1)} / {number(values['decode_tokens'], 1)} | "
                         f"{number(values['total_engine_wall_s'])} | {number(values['weight_copy_bytes'] / 1e9)} | "
                         f"{number(values['groups'], 1)} |")
    lines.extend(['', '| 臂均值 | 专家 bytes/输出 token | KV load / store MB | 抢占 | scheduled draft rows |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for group, values in summary['repeat_means'].items():
        kv = [number(values[f'kv_{key}_bytes'] / 1e6 if values[f'kv_{key}_bytes'] is not None else None)
              for key in ('load', 'store')]
        lines.append(f"| {group} | {number(values['expert_bytes_per_output_token'], 0)} | {' / '.join(kv)} | "
                     f"{number(values['preemptions'], 1)} | {number(values['scheduled_draft_token_rows'], 1)} |")
    lines.extend(['', 'GB/MB 使用十进制。没有把专家传输字节另折算并叠加到 engine 时间。', '',
                  '结束状态：' + '；'.join(f'{name}={counts}' for name, counts in summary['cell_finish_counts'].items()) + '。', '',
                  *[f'- {note}' for note in summary['notes']], ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--design', choices=tuple(DESIGNS), default='ngram_short')
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs/olmoe_gsm8k_natural16')
    parser.add_argument('--metadata', type=Path,
                        default=ROOT.parent / 'C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1')
    parser.add_argument('--existing-analysis', action='store_true',
                        help='复用 results 中现有 metrics.json 和 ngram_activity.json，不再次运行 analyzer')
    parser.add_argument('--output', type=Path, help='JSON 路径；默认 results/ngram_short_summary.json')
    parser.add_argument('--report', type=Path, help='中文 Markdown 路径；默认与 JSON 同名 .md')
    args = parser.parse_args()
    groups = DESIGNS[args.design]
    try:
        metrics, activity = read_analysis(args, groups)
        summary = build_summary(metrics, activity, groups, args.results, args.design)
        report = render_report(summary)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    output = args.output or args.results / f'{args.design}_summary.json'
    report_path = args.report or output.with_suffix('.md')
    if output.resolve() == report_path.resolve():
        parser.error('JSON 与中文报告路径不能相同')
    protected = {args.results.resolve() / name for name in ('metrics.json', 'ngram_activity.json', 'group_status.json')}
    if output.resolve() in protected or report_path.resolve() in protected:
        parser.error('请使用独立汇总路径，不覆盖既有分析/组状态')
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report_path.write_text(report, encoding='utf-8')
    print(report)
    print(f'JSON: {output}\n中文报告: {report_path}')


if __name__ == '__main__':
    main()
