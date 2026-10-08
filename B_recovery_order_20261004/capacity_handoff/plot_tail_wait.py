#!/usr/bin/env python3
"""Plot the sole observed post-LOAD-ack allocation-failure episode, not a speedup."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=root/'session-20261007-r01')
    parser.add_argument('--output', type=Path, default=root/'tail-capacity-wait.png')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads((args.session/'tail-allocation-analysis.json').read_text())
    episodes = [e for e in data['episodes'] if e['post_ack_failures']]
    if len(episodes) != 1:
        raise ValueError('This figure requires exactly one observed failure episode')
    episode = episodes[0]
    attempts = [a for a in episode['attempts'] if a['phase'] == 'after_load_ack']
    loads = episode['load_jobs']
    if len(loads) != 1:
        raise ValueError('This scoped figure needs one LOAD in the episode')
    ack = loads[0]['times']['ack_retired']
    before = next(a['record'] for a in episode['attempts']
                  if a['phase'] == 'before_load_ack' and a['record']['success'])
    success = next(a['record'] for a in attempts if a['record']['success'])
    failed = [a['record'] for a in attempts if not a['record']['success']]
    if not all(r['descriptor']['exact'] for r in failed+[success]):
        raise ValueError('Unsupported capacity descriptor')
    slot = {r['descriptor']['slot_extra_blocks'] for r in failed+[success]}
    full = {r['descriptor']['full_fit_extra_blocks'] for r in failed+[success]}
    held = {tuple(r['before']['held_gpu_blocks']) for r in failed+[success]}
    if len(slot) != 1 or len(full) != 1 or len(held) != 1:
        raise ValueError('Requirements/counts vary; this figure must not flatten them')
    slot, full, held = slot.pop(), full.pop(), held.pop()[0]
    cell = args.session/'cell-00-cap256-native/output'
    raw = json.loads((cell/'raw.json').read_text())
    origin = raw['measurement_origin_perf_counter_s']
    request = next(r for r in raw['requests'] if r['request_id'] == episode['source_request'])
    output = next(t+origin for t in request['token_times_s'] if t > episode['begin_s'])
    start = episode['begin_host_perf_s']
    initial = before['begin_host_perf_s']
    allocated = success['begin_host_perf_s']
    bounds = [start, initial, ack, allocated, output]
    if bounds != sorted(bounds):
        raise ValueError('Nonmonotone host boundaries')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, (ax, chain) = plt.subplots(2, 1, figsize=(11.3, 7.3),
                                  gridspec_kw={'height_ratios': [3.3, 1.15]})
    for r in failed:
        ms = (r['begin_host_perf_s']-ack)*1000
        free = r['before']['free_gpu_blocks']
        color = '#C47A00' if free >= slot else '#B33A3A'
        ax.scatter(ms, free, c=color, marker='x', s=66, linewidths=2, zorder=4)
        ax.annotate(str(free), (ms, free), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10, color=color)
    ms = (allocated-ack)*1000
    free = success['before']['free_gpu_blocks']
    ax.scatter(ms, free, c='#14805E', marker='o', s=65, zorder=4)
    ax.annotate(f'{free} free; allocation succeeds', (ms, free), xytext=(-8, 13),
                textcoords='offset points', ha='right', color='#126C50', fontsize=10)
    ax.axhline(full, color='#B33A3A', ls='--', lw=1.5)
    ax.axhline(slot, color='#247CA5', ls=':', lw=1.8)
    ax.set(xlim=(-25, 700), ylim=(0, 220), xlabel='Host time since LOAD acknowledgement (ms)',
           ylabel='Free GPU blocks before allocation attempt')
    ax.set_title(f'(a) {held} prefix blocks held; {len(failed)} attempts fail before the next succeeds',
                 loc='left', pad=11, fontsize=12)
    ax.grid(axis='y', color='#E6E6E6', lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[Line2D([], [], color='#B33A3A', ls='--', label=f'Full-history fit requirement: {full} extra blocks'),
                       Line2D([], [], color='#247CA5', ls=':', label=f'Current compute chunk: {slot} extra blocks')],
              loc='upper left', fontsize=9, frameon=False)

    names = ['Demand to prefix\nallocation attempt', 'Prefix allocation\nto LOAD ACK',
             'ACK to successful\ntail allocation attempt', 'Tail allocation\nto next output']
    colors = ['#CDD6DD', '#89BAD4', '#EABF7A', '#8DC6AD']
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        left, width = (lo-ack)*1000, (hi-lo)*1000
        chain.barh(0, width, left=left, height=.44, color=colors[i], edgecolor='white')
        chain.text(left+width/2, 0, f'{names[i]}\n{width:.1f} ms', ha='center', va='center', fontsize=9)
    chain.axvline(0, color='#414141', lw=.8)
    chain.set_xlim((start-ack)*1000-12, (output-ack)*1000+12)
    chain.set_ylim(-.38, .38)
    chain.set_yticks([])
    chain.set_xlabel('Host time since LOAD acknowledgement (ms)')
    chain.set_title('(b) Observed recovery-to-next-output chain', loc='left', fontsize=12, pad=10)
    chain.spines[['top', 'right', 'left']].set_visible(False)
    fig.suptitle('After LOAD acknowledgement, tail allocation still waits',
                 x=.085, ha='left', y=.974, fontsize=17, fontweight='bold')
    fig.text(.085, .916, 'Native run: 256/256 requests complete. All seven post-ACK failures occur in this one recovery episode.', fontsize=10)
    fig.text(.085, .04,
             f'Observed case: {episode["source_request"]}, LOAD {loads[0]["job_id"]}. Each block = 16 tokens / 2 MiB.\n'
             f'Before LOAD: {before["before"]["free_gpu_blocks"]} free, full-fit checks {before["descriptor"]["full_fit_extra_blocks"]}, '
             f'but only {before["after"]["held_gpu_blocks"][0]} prefix blocks are allocated. Points are discrete samples, not a continuous free-capacity trace.\n'
             'All times are host observations; ACK is not the GPU completion instant. No intervention or counterfactual speedup is shown.',
             fontsize=8.3, linespacing=1.55)
    fig.subplots_adjust(left=.085, right=.975, top=.85, bottom=.19, hspace=.58)
    with args.output.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'post_ack_failures': len(failed),
                      'ack_to_tail_allocation_ms': (allocated-ack)*1000,
                      'ack_to_next_output_ms': (output-ack)*1000}))


if __name__ == '__main__':
    main()
