"""Six run-level panels from analyze_suffix_group.py's suffix_metrics.json.

Example:
  /private/tmp/moe-c-plot-env/bin/python plot_suffix_results.py \
    --metrics execution_suffix_41307_r02/results_suffix_r02/suffix_metrics.json \
    --output-prefix figures/suffix_r02_service

Arms come from config (ngram_speculative_tokens, suffix_policy, suffix_compact),
or expert_cap/kv_bytes with --group-by partition, never directory names.
Missing/incomplete observations are not plotted as zero.
No raw analysis, confidence intervals, or GPU execution is performed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import textwrap

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'moe-serving-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


COLORS = ('#0072B2', '#009E73', '#D55E00', '#CC79A7', '#E69F00', '#56B4E9')
MARKERS = ('o', 'D', '^', 's', 'v', 'P')
FIELDS = (
    ('drained_s', '(a) Complete service time', 'Seconds', ('kv_service_cost', 'drained_s'), 1, '.2f'),
    ('token_rate', '(b) Delivered output rate', 'Returned tokens / drained second', ('drained_output_tokens_per_s',), 1, '.2f'),
    ('mean_flow_s', '(c) Mean request flow', 'Seconds', ('flow', 'mean_s'), 1, '.2f'),
    ('output_tokens', '(d) Returned output', 'Tokens', ('output_tokens',), 1, ',.0f'),
    ('expert_gb', '(e) Total expert H2D traffic', 'GB (10$^9$ bytes)', ('weight_copy_bytes',), 1e9, '.1f'),
    ('correct_requests', '(f) Correct requests', 'Requests', ('correct_requests',), 1, '.1f'),
)


def arm_key(cell, group_by='method'):
    config = cell.get('config') or {}
    if group_by == 'partition':
        cap, kv = config.get('expert_cap'), config.get('kv_bytes')
        if any(type(value) is not int or value <= 0 for value in (cap, kv)):
            raise ValueError('Partition grouping requires positive expert_cap and kv_bytes')
        return cap, kv
    n = config.get('ngram_speculative_tokens')
    policy = config.get('suffix_policy', 'off')
    compact = config.get('suffix_compact', False)  # Legacy captures predate the flag.
    if type(n) is not int or n < 0 or not isinstance(policy, str) or type(compact) is not bool:
        raise ValueError('Each cell needs valid ngram_speculative_tokens/policy/compact configuration')
    return n, policy, compact


def arm_label(key, group_by='method'):
    if group_by == 'partition':
        cap, kv = key
        return f'Expert {cap}\nKV {kv / 1024**3:g} GiB'
    n, policy, compact = key
    if n == 0 and policy == 'off' and not compact:
        return 'AR'
    return f'N{n}\n' + ('no cut' if policy == 'off' else policy) + ('\n+ compact' if compact else '')


def read_data(source, group_by='method'):
    cells, arms = source['cells'], {}
    names = list(dict.fromkeys(list(source.get('order', [])) + list(cells)))
    for name in names:
        cell = cells[name]
        key = arm_key(cell, group_by)
        metrics = cell.get('metrics') or {}
        complete = (cell.get('analysis_status') == 'COMPLETE'
                    and metrics.get('all_16_complete', metrics.get('all_complete', False)))
        values = {}
        for field, _, _, path, scale, _ in FIELDS:
            value = metrics
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            valid = (complete and type(value) in (int, float)
                     and math.isfinite(value) and value >= 0)
            if field == 'drained_s' and valid:
                valid = value > 0
            values[field] = value / scale if valid else None
        arms.setdefault(key, []).append(dict(name=name, complete=bool(complete), values=values,
                                             metrics=metrics, config=cell.get('config') or {}))
    if not arms or len(arms) > len(COLORS):
        raise ValueError('Expected one to six configured arms')
    return dict(sorted(arms.items(), key=lambda item: (-item[0][0], item[0][1])
                       if group_by == 'partition' else item[0]))


def caption_for(arms, domain_note, group_by='method'):
    runs = [run for runs in arms.values() for run in runs]
    complete = [r for r in runs if r['complete']]
    caps = {r['config'].get('expert_cap') for r in runs}
    requests = {r['metrics'].get('requests') for r in complete}
    sizes = str(next(iter(requests))) if len(requests) == 1 and None not in requests else 'varying numbers of'
    if domain_note is None and group_by == 'partition':
        totals = set()
        for run in complete:
            metrics = run['metrics']
            scratch = metrics.get('expert_scratch_bytes')
            kv = metrics.get('actual_resources', {}).get('before_measurement', {}).get('gpu_kv_unique_bytes')
            totals.add(scratch + kv if all(type(value) is int and value > 0 for value in (scratch, kv)) else None)
        if len(totals) == 1 and None not in totals:
            total = next(iter(totals)) / 1024**3
            domain_note = f'OLMoE; equal {total:g} GiB expert + GPU KV pools (artificial budget; excludes other allocations).'
        else:
            domain_note = 'OLMoE; artificial expert + GPU KV pool partitions; equal total size is not established by available measurements.'
    if domain_note is None:
        if caps != {24}:
            raise ValueError('Supply --domain-note outside the known artificial expert24 domain')
        domain_note = 'OLMoE; artificial expert-cache cap of 24 experts/layer.'
    counts = {r['metrics'].get('output_tokens') for r in complete}
    output_note = ('Actual output counts differ; times are not equal-work speedups.' if len(counts) > 1
                   else 'Equal output counts do not establish identical output content.')
    incomplete = len(runs) - len(complete)
    missing = sum(v is None for r in complete for v in r['values'].values())
    omissions = (f' {incomplete} incomplete runs excluded; {missing} missing metric values omitted.'
                 if incomplete or missing else '')
    eos = ('Natural EOS' if complete and all(r['config'].get('natural_eos')
           and r['metrics'].get('cap_truncated_requests') == 0 for r in complete)
           else 'Observed generation stopping rules')
    return (f'{eos}; {sizes} requests/run. Bars: available-run means; points: individual complete runs '
            '(ordered within arm). Run-level n is shown per panel; small samples, no CI. '
            'Drained time includes final drain and excludes initialization/warmup. '
            f'{output_note} {domain_note} Host chunk gaps are not token-level ITL.{omissions}')


def plot(source, output_prefix, domain_note=None, group_by='method'):
    arms = read_data(source, group_by)
    caption = caption_for(arms, domain_note, group_by)
    paths = [output_prefix.with_suffix('.' + extension) for extension in ('svg', 'pdf', 'png')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Output exists; choose a new --output-prefix to preserve previous figures')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.35))
    report = {}
    for ax, (field, title, unit, _, _, format_spec) in zip(axes.flat, FIELDS):
        all_values = [r['values'][field] for runs in arms.values() for r in runs if r['values'][field] is not None]
        upper = max(all_values, default=1)
        if field == 'correct_requests':
            upper = max([upper] + [r['metrics'].get('requests') or 0 for runs in arms.values() for r in runs])
        upper = max(upper * 1.20, 1)
        labels = []
        for index, (key, runs) in enumerate(arms.items()):
            observations = [(j, r['values'][field]) for j, r in enumerate(runs) if r['values'][field] is not None]
            values = [value for _, value in observations]
            labels.append(arm_label(key, group_by) + f'\nn={len(values)}')
            report.setdefault(str(key), {'cells': [r['name'] for r in runs], 'metrics': {}})['metrics'][field] = {
                'n': len(values), 'mean': statistics.mean(values) if values else None, 'runs': values}
            if values:
                mean = statistics.mean(values)
                ax.bar(index, mean, width=0.58, color=COLORS[index], alpha=0.68,
                       edgecolor=COLORS[index], linewidth=0.8, zorder=2)
                for j, value in observations:
                    jitter = (j - (len(runs) - 1) / 2) * min(0.13, 0.45 / max(1, len(runs) - 1))
                    ax.scatter(index + jitter, value, s=29, marker=MARKERS[j % len(MARKERS)],
                               facecolor='white', edgecolor='#242424', linewidth=0.85, zorder=4)
                ax.text(index, max(values) + upper * 0.035, format(mean, format_spec),
                        ha='center', va='bottom', fontsize=8, color='#242424')
            else:
                ax.text(index, upper * 0.10, 'No data', ha='center', va='bottom',
                        fontsize=8, color='#777777')
        ax.set_title(title, loc='left', pad=9, fontweight='semibold')
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(arms)), labels)
        ax.set_xlim(-0.65, len(arms) - 0.35)
        ax.set_ylim(0, upper)
        ax.tick_params(axis='x', length=0, pad=6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=field in ('correct_requests', 'output_tokens')))
        ax.grid(axis='y', alpha=0.23, linewidth=0.5)
        ax.set_axisbelow(True)
    max_repeats = max(map(len, arms.values()))
    handles = [Line2D([], [], color='#242424', marker=MARKERS[j % len(MARKERS)],
               markerfacecolor='white', linestyle='None', markersize=5, label=f'Run {j + 1}')
               for j in range(max_repeats)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.51, 0.991),
               ncol=min(max_repeats, 6), frameon=False)
    fig.text(0.06, 0.025, '\n'.join(textwrap.wrap(caption, 165)), fontsize=8,
             ha='left', va='bottom', linespacing=1.35, color='#404040')
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.235, hspace=0.78, wspace=0.34)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        metadata = {'Title': 'Complete service and delivered output', 'Creator': 'plot_suffix_results.py'}
        if path.suffix == '.pdf':
            metadata['Subject'] = caption
        elif path.suffix == '.svg':
            metadata['Description'] = caption
        fig.savefig(path, dpi=240, facecolor='white', metadata=metadata)
        print(path)
    plt.close(fig)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--group-by', choices=('method', 'partition'), default='method',
                        help='Group by draft method (default), or expert_cap/kv_bytes')
    parser.add_argument('--domain-note', help='Override the automatic expert24 or pool-partition domain caption')
    args = parser.parse_args()
    try:
        plot(json.loads(args.metrics.read_text(encoding='utf-8')), args.output_prefix, args.domain_note, args.group_by)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
