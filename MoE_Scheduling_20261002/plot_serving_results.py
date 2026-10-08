"""Plot actual run-level MoE serving tradeoffs from existing metrics.json.

Example:
  MPLCONFIGDIR=/private/tmp/moe-serving-mpl /private/tmp/moe-c-plot-env/bin/python \
    plot_serving_results.py --metrics results_ngram_r01/metrics.json \
    --output-prefix figures/motivation_ngram_r01

AR/N1/N2 names are inferred for new ngram groups. For other naming schemes,
repeat --group LABEL=cell1,cell2 and set --reference LABEL. No raw reanalysis,
bootstrap, token-level confidence interval, or weight download is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import tempfile
import textwrap

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'moe-serving-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
MARKERS = ('o', 'D', '^', 's', 'v')
METRICS = (
    ('decode_steps', 'Pure-decode\nsteps', lambda cell: cell['stages']['by_kind']['pure_decode']['scheduler_steps']),
    ('expert_bytes', 'Total expert\nH2D bytes', lambda cell: cell['weight_copy_bytes']),
    ('complete_time', 'Completion\ntime', lambda cell: cell['kv_service_cost']['drained_s']),
    ('output_tokens', 'Returned\ntokens', lambda cell: cell['output_tokens']),
    ('output_rate', 'Returned tokens\n/ completion time', lambda cell: cell['drained_output_tokens_per_s']),
)


def resolve_groups(metrics, custom):
    if custom:
        groups = {}
        for definition in custom:
            label, separator, raw_names = definition.partition('=')
            names = [value.strip() for value in raw_names.split(',') if value.strip()]
            if not separator or not label.strip() or not names or label.strip() in groups:
                raise ValueError('--group must be a unique LABEL=cell1,cell2')
            groups[label.strip()] = names
        return groups
    groups = {}
    for name in metrics.get('order', metrics['cells']):
        if re.search(r'_ar\d*$', name):
            label = 'AR'
        else:
            match = re.search(r'_ngram(\d+)$', name)
            if match is None:
                raise ValueError(f'Cannot infer arm for {name}; supply --group LABEL=cell1,cell2')
            label = 'N' + match.group(1)
        groups.setdefault(label, []).append(name)
    return groups


def read_values(metrics, groups):
    data = {}
    for group, names in groups.items():
        data[group] = {}
        for name in names:
            cell = metrics['cells'][name]
            if not cell.get('all_16_complete', cell.get('all_complete', False)):
                raise ValueError(f'{name}: incomplete runs cannot support complete-time comparison')
        for key, _, getter in METRICS:
            values = [getter(metrics['cells'][name]) for name in names]
            if any(value is None or value <= 0 for value in values):
                raise ValueError(f'{group}/{key}: missing or nonpositive measurement')
            data[group][key] = values
    return data


def make_caption(metrics, groups, domain_note):
    cells = [metrics['cells'][name] for names in groups.values() for name in names]
    sizes = {cell['requests'] for cell in cells}
    repeats = {len(names) for names in groups.values()}
    count = str(next(iter(sizes))) if len(sizes) == 1 else 'varying'
    n = str(next(iter(repeats))) if len(repeats) == 1 else 'varying'
    caps = {cell.get('expert_cap') for cell in cells}
    if domain_note is None:
        if len(caps) != 1 or None in caps:
            raise ValueError('Specify --domain-note when expert-cap metadata are missing or unequal')
        domain_note = f'OLMoE; artificial expert-cache cap of {next(iter(caps))} experts/layer.'
    any_caps = any(cell.get('cap_truncated_requests', 0) for cell in cells)
    eos_note = 'Natural-EOS contract, with length-capped outputs' if any_caps else 'Natural EOS'
    # Output-count comparison is independent of text identity; the JSON retains
    # full sequences, and the caption explicitly disallows equal-work inference.
    same_counts = len({cell['output_tokens'] for cell in cells}) == 1
    work_note = ('Output counts differ; times are not equal-work speedups.' if not same_counts
                 else 'Equal token counts alone do not establish identical output content.')
    return (f'{eos_note}; {count} requests/run, n={n} independent runs/arm. '
            'Bars: run means; points: individual runs. '
            'Each metric is normalized by the AR run mean. '
            'Completion time includes final drain; initialization and warmup are excluded. '
            f'{work_note} {domain_note}')


def plot(metrics, groups, reference, output_prefix, domain_note):
    if reference not in groups or len(groups) < 2:
        raise ValueError('A reference arm and at least one other arm are required')
    if len(groups) > len(COLORS):
        raise ValueError('Keep this compact figure to at most five arms')
    data = read_values(metrics, groups)
    order = [reference] + [group for group in groups if group != reference]
    means = {group: {key: statistics.mean(values) for key, values in fields.items()}
             for group, fields in data.items()}
    normalized = {group: {key: [value / means[reference][key] for value in values]
                          for key, values in fields.items()} for group, fields in data.items()}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.75),
                                gridspec_kw={'width_ratios': [3, 2]}, sharey=True)
    width = min(0.28, 0.76 / len(order))
    maximum = max(value for fields in normalized.values() for values in fields.values() for value in values)
    ylim = max(1.36, maximum * 1.17)
    for axis, subset, title in zip(axes, (METRICS[:3], METRICS[3:]),
                                   ('(a) Work and complete service cost', '(b) Delivered output')):
        axis.axhline(1, color='#777777', linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        annotation_positions = {}
        for group_index, group in enumerate(order):
            shift = (group_index - (len(order) - 1) / 2) * width
            xs = [index + shift for index in range(len(subset))]
            heights = [means[group][key] / means[reference][key] for key, _, _ in subset]
            axis.bar(xs, heights, width=width * 0.86, color=COLORS[group_index], alpha=0.72,
                     edgecolor=COLORS[group_index], linewidth=0.9, zorder=2)
            for x, (key, _, _), height in zip(xs, subset, heights):
                values = normalized[group][key]
                for run_index, value in enumerate(values):
                    jitter = (run_index - (len(values) - 1) / 2) * width * 0.19
                    axis.scatter(x + jitter, value, s=17, marker=MARKERS[run_index % len(MARKERS)],
                                 facecolor='white', edgecolor='#242424', linewidth=0.7, zorder=4)
                if group != reference:
                    label_y = max(values) + ylim * 0.036
                    previous_y = annotation_positions.get(key)
                    if previous_y is not None and abs(label_y - previous_y) < ylim * 0.08:
                        label_y = previous_y + ylim * 0.09
                        axis.plot([x, x], [max(values) + ylim * 0.016, label_y - ylim * 0.012],
                                  color=COLORS[group_index], linewidth=0.5, alpha=0.55, zorder=3)
                    annotation_positions[key] = label_y
                    axis.text(x, label_y, f'{(height - 1) * 100:+.1f}%',
                              ha='center', va='bottom', fontsize=8 if len(order) == 2 else 7,
                              color=COLORS[group_index], fontweight='bold')
        axis.set_xticks(range(len(subset)), [label for _, label, _ in subset])
        axis.tick_params(axis='x', length=0, pad=7)
        axis.set_title(title, loc='left', pad=10, fontweight='semibold')
        axis.set_ylim(0, ylim)
        axis.set_yticks([0, 0.5, 1.0] + ([1.5] if ylim >= 1.6 else []))
        axis.grid(axis='y', linewidth=0.45, alpha=0.22, zorder=0)
        axis.set_axisbelow(True)
    axes[0].set_ylabel(f'Normalized to {reference} mean')
    handles = [Patch(facecolor=COLORS[index], alpha=0.72, edgecolor=COLORS[index], label=group)
               for index, group in enumerate(order)]
    handles += [Line2D([], [], color='#242424', marker=marker, markerfacecolor='white',
                       linestyle='None', markersize=4, label=f'Run {index + 1}')
                for index, marker in enumerate(MARKERS[:max(map(len, groups.values()))])]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.985),
                  ncol=len(handles), frameon=False, handlelength=1.3, columnspacing=1.4)
    caption = make_caption(metrics, groups, domain_note).replace('by the AR run mean', f'by the {reference} run mean')
    figure.text(0.065, 0.025, '\n'.join(textwrap.wrap(caption, 128)),
                ha='left', va='bottom', fontsize=7.1, linespacing=1.35, color='#404040')
    figure.subplots_adjust(left=0.083, right=0.985, top=0.80, bottom=0.34, wspace=0.10)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'pdf', 'png'):
        path = output_prefix.with_suffix('.' + extension)
        metadata = {'Title': 'Fixed n-gram speculation: fewer steps, higher expert traffic',
                    'Creator': 'plot_serving_results.py'}
        if extension == 'svg':
            metadata['Description'] = caption
        elif extension == 'pdf':
            metadata['Subject'] = caption
        figure.savefig(path, dpi=240, metadata=metadata)
        print(path)
    plt.close(figure)
    print(json.dumps({'run_means': means, 'reference': reference,
                      'ratios': {group: {key: means[group][key] / means[reference][key]
                                          for key in means[group]} for group in order}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT / 'results_ngram_r01/metrics.json')
    parser.add_argument('--output-prefix', type=Path, default=ROOT / 'figures/motivation_ngram_r01')
    parser.add_argument('--group', action='append', help='LABEL=cell1,cell2; repeat for each arm')
    parser.add_argument('--reference', default='AR')
    parser.add_argument('--domain-note', help='Override OLMoE/artificial-cap caption for another deployment domain')
    args = parser.parse_args()
    try:
        metrics = json.loads(args.metrics.read_text(encoding='utf-8'))
        plot(metrics, resolve_groups(metrics, args.group), args.reference, args.output_prefix, args.domain_note)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
