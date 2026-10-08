#!/usr/bin/env python3
"""Plot explicit four-arm admission comparisons without overwriting outputs."""
import argparse
import json
import math
from pathlib import Path

from compare_quartet import validate_matched_calibration
from plot_comparison import FIELDS, Line2D, observed_cdf, plt


BLOCKS = (('probe-00-baseline', 'probe-01-delay'),
          ('probe-03-baseline', 'probe-02-delay'))
POLICIES = (('KV baseline', '#0072B2', '-'),
            ('100 ms one-shot delay', '#D55E00', (0, (4, 2))))
PROGRESS_BLOCKS = (('probe-00-timer', 'probe-01-progress'),
                   ('probe-03-timer', 'probe-02-progress'))
PROGRESS_POLICIES = (('100 ms timer', '#0072B2', '-'),
                     ('Recovery progress release', '#D55E00', (0, (4, 2))))
RESERVATION_BLOCKS = (('probe-00-baseline', 'probe-01-reservation'),
                      ('probe-03-baseline', 'probe-02-reservation'))
RESERVATION_POLICIES = (('KV baseline', '#0072B2', '-'),
                        ('Known-prefill reservation', '#D55E00', (0, (4, 2))))
HEADROOM_BLOCKS = (('probe-00-baseline', 'probe-01-headroom32'),
                   ('probe-03-baseline', 'probe-02-headroom32'))
HEADROOM_POLICIES = (('KV baseline', '#0072B2', '-'),
                     ('Fixed 32-page headroom', '#D55E00', (0, (4, 2))))
SIMPLE_BLOCKS = (('probe-00-fixed192', 'probe-01-kv256'),
                 ('probe-03-fixed192', 'probe-02-kv256'))
SIMPLE_POLICIES = (('Fixed192 (KV floor 0)', '#0072B2', '-'),
                   ('KV256 (floor 3277; 10 s age bypass)', '#D55E00', (0, (4, 2))))
DECLARED_BLOCKS = (('probe-00-fixed128', 'probe-01-declaredbudget'),
                   ('probe-03-fixed128', 'probe-02-declaredbudget'))
DECLARED_POLICIES = (('Fixed cap 128', '#0072B2', '-'),
                     ('Declared budget: 32768 pages / cap 256', '#D55E00', (0, (4, 2))))
DECLARED_MATCHED_BLOCKS = (('probe-00-fixed177', 'probe-01-declaredbudget'),
                          ('probe-03-fixed177', 'probe-02-declaredbudget'))
DECLARED_MATCHED_POLICIES = (('Fixed cap 177 (development mean-charge calibration)', '#0072B2', '-'),
                            ('Declared budget: 32768 pages / cap 256', '#D55E00', (0, (4, 2))))
MC_BLOCKS = (('probe-00-declaredbudget', 'probe-01-mcbudget'),
             ('probe-03-declaredbudget', 'probe-02-mcbudget'))
MC_POLICIES = (('Declared lifetime sum', '#0072B2', '-'),
               ('MC-Benchmark (guarded adaptation)', '#D55E00', (0, (4, 2))))
MC_ABLATION_BLOCKS = (('probe-00-mcstatic', 'probe-01-mccapped'),
                      ('probe-03-mcstatic', 'probe-02-mccapped'))
MC_ABLATION_POLICIES = (('Static oversubscription', '#0072B2', '-'),
                        ('Future-peak capped', '#D55E00', (0, (4, 2))))
MC_PHASE_BLOCKS = (('probe-00-mccapped', 'probe-01-mcphase'),
                   ('probe-03-mccapped', 'probe-02-mcphase'))
MC_PHASE_POLICIES = (('Strict decode phase', '#0072B2', '-'),
                     ('Scheduled-prefill completion allowed', '#D55E00', (0, (4, 2))))


def select_layout(cells):
    """Accept only explicit arm labels; never infer a no-intervention baseline."""
    matches = [blocks for blocks in (BLOCKS, PROGRESS_BLOCKS, RESERVATION_BLOCKS, HEADROOM_BLOCKS,
                                    SIMPLE_BLOCKS, DECLARED_BLOCKS, DECLARED_MATCHED_BLOCKS,
                                    MC_BLOCKS, MC_ABLATION_BLOCKS, MC_PHASE_BLOCKS)
               if all(name in cells for block in blocks for name in block)]
    if len(matches) != 1:
        raise ValueError('Expected one explicit baseline/delay/delay/baseline or '
                         'timer/progress/progress/timer, reservation ABBA, fixed-headroom ABBA, '
                         'fixed192/KV256 quartet, fixed128 or fixed177/declared-budget quartet, '
                         'declared-budget/MC-budget quartet, static/capped quartet, or capped/phase quartet.')
    blocks = matches[0]
    progress = blocks == PROGRESS_BLOCKS
    reservation = blocks == RESERVATION_BLOCKS
    headroom = blocks == HEADROOM_BLOCKS
    simple = blocks == SIMPLE_BLOCKS
    matched = blocks == DECLARED_MATCHED_BLOCKS
    declared = blocks in (DECLARED_BLOCKS, DECLARED_MATCHED_BLOCKS)
    mc = blocks == MC_BLOCKS
    mc_ablation = blocks == MC_ABLATION_BLOCKS
    mc_phase = blocks == MC_PHASE_BLOCKS
    for block in blocks:
        for name in block:
            cell = cells[name]
            configurations = (cell.get('config', {}), cell.get('configuration', {}),
                              cell.get('admission', {}),
                              cell.get('admission', {}).get('configuration', {}))
            if simple or declared or mc or mc_ablation or mc_phase:
                fixed = name.endswith(('-fixed192', '-fixed128', '-fixed177'))
                cap = (177 if matched else 128 if declared else 192) if fixed else 256
                expected_fields = [('cap', cap), ('kv_floor', 0 if fixed or declared or mc or mc_ablation or mc_phase else 3277),
                                   ('probe_enabled', False)]
                if simple:
                    expected_fields.append(('max_signal_wait_s', 10))
                if (declared and not fixed) or mc or mc_ablation or mc_phase:
                    expected_fields += [('budget_blocks', 32768), ('block_size', 16),
                                        ('max_signal_wait_s', None)]
                    if not name.endswith(('-mcbudget', '-mcstatic', '-mccapped', '-mcphase')):
                        expected_fields.append(('budget_overrides', 0))
                    else:
                        overrides = [config['budget_overrides'] for config in configurations
                                     if 'budget_overrides' in config]
                        if not overrides or any(type(value) is not int or value < 0 for value in overrides):
                            raise ValueError(f'{name}: budget_overrides must be a recorded nonnegative count')
                for key, expected in expected_fields:
                    values = [config[key] for config in configurations if key in config]
                    if not values or any(value != expected or
                            (key == 'probe_enabled' and value is not False) for value in values):
                        raise ValueError(f'{name}: missing or conflicting {key}; expected {expected!r}')
                # The installed no-probe gate reports mode=kv even at floor zero.
                # Validate each arm's actual limits, without equating their caps.
                for config in configurations:
                    for key in ('mode', 'admission_mode'):
                        allowed_modes = (('mc_budget_static',) if name.endswith('-mcstatic') else
                                         ('mc_budget_phase',) if name.endswith('-mcphase') else
                                         ('mc_budget_capped',) if name.endswith('-mccapped') else
                                         ('mc_budget',) if name.endswith('-mcbudget') else
                                         ('fixed', 'kv') if fixed else
                                         ('declared_budget',) if declared or mc else ('kv',))
                        if key in config and config[key] not in allowed_modes:
                            raise ValueError(f'{name}: {key} conflicts with the simple-rule label')
                    if 'admission_cap' in config and config['admission_cap'] != cap:
                        raise ValueError(f'{name}: admission_cap conflicts with its explicit arm label')
                    for key in ('reservation_probe_enabled', 'reservation_observation'):
                        if key in config and config[key] is not False:
                            raise ValueError(f'{name}: {key} must be disabled for the simple-rule comparison')
                continue
            for config in configurations:
                if not config:
                    continue
                expected_enabled = not (reservation or headroom) and not name.endswith('-baseline')
                if 'probe_enabled' in config and config['probe_enabled'] != expected_enabled:
                    raise ValueError(f'{name}: probe_enabled conflicts with its explicit arm label')
                if (reservation or headroom) and 'reservation_probe_enabled' in config:
                    if config['reservation_probe_enabled'] != (not name.endswith('-baseline')):
                        raise ValueError(f'{name}: allocation probe conflicts with its explicit arm label')
                if (reservation or headroom) and 'reservation_signal_kind' in config:
                    if config['reservation_signal_kind'] != ('fixed_headroom' if headroom else 'known_prefill'):
                        raise ValueError(f'{name}: signal kind conflicts with its explicit arm label')
                if headroom and 'fixed_headroom_blocks' in config and config['fixed_headroom_blocks'] != 32:
                    raise ValueError(f'{name}: fixed headroom conflicts with the 32-page label')
                expected_mode = 'output_progress' if name.endswith('-progress') else 'timer'
                if 'release_mode' in config and config['release_mode'] != expected_mode:
                    raise ValueError(f'{name}: release_mode conflicts with its explicit arm label')
                if progress:
                    for key, expected in (('delay_s', .1), ('max_extra_s', .25),
                                          ('hard_gate_deadline_s', .25)):
                        if key in config and not math.isclose(config[key], expected, abs_tol=1e-9):
                            raise ValueError(f'{name}: {key} conflicts with the 100/250 ms caption')
                if reservation or headroom:
                    for key, expected in (('reservation_min_hold_s', 0), ('hard_gate_deadline_s', .25)):
                        if key in config and not math.isclose(config[key], expected, abs_tol=1e-9):
                            raise ValueError(f'{name}: {key} conflicts with the allocation-probe caption')
    policies = (MC_PHASE_POLICIES if mc_phase else MC_ABLATION_POLICIES if mc_ablation else MC_POLICIES if mc else DECLARED_MATCHED_POLICIES if matched else DECLARED_POLICIES if declared else SIMPLE_POLICIES if simple else HEADROOM_POLICIES if headroom else RESERVATION_POLICIES if reservation else
                PROGRESS_POLICIES if progress else POLICIES)
    return blocks, policies, progress


def native_guard_recheck(cells, names, cell_root=None):
    """Recognize only the explicitly recorded four-arm native-guard repair."""
    matched = any(name.endswith('-fixed177') for name in names)
    mc_phase = any(name.endswith('-mcphase') for name in names)
    configs = []
    for name in names:
        cell = cells[name]
        config = cell.get('config')
        if config is None:
            path = (cell_root / name if cell_root is not None else Path(cell['cell'])) / 'config.json'
            config = json.loads(path.read_text()) if path.is_file() else {}
        if matched:
            validate_matched_calibration(config)
        configs.append(config)
    flags = [config.get('native_admission_guard_matches_complete_cap', False) for config in configs]
    if not any(flags):
        return False
    if not all(flag is True for flag in flags):
        raise ValueError('Native-guard repair must be explicitly recorded for all four arms.')
    for name, config in zip(names, configs):
        cap = 177 if name.endswith('-fixed177') else 128 if name.endswith('-fixed128') else 192 if name.endswith('-fixed192') else 256
        expected = dict(cap=cap, admission_cap=cap, native_running_limit=cap,
                        engine_max_num_seqs=256, admission_count='complete_unique_unfinished')
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f'{name}: {key} must be {value!r} for the native-guard caption')
        if config.get('probe_enabled') is not False:
            raise ValueError(f'{name}: native-guard recheck must retain the no-probe complete Gate')
        if name.endswith(('-fixed128', '-fixed177', '-declaredbudget', '-mcbudget', '-mcstatic', '-mccapped', '-mcphase')):
            fixed = name.endswith(('-fixed128', '-fixed177'))
            expected_budget = dict(admission_mode='fixed' if fixed else
                'mc_budget_static' if name.endswith('-mcstatic') else
                'mc_budget_phase' if name.endswith('-mcphase') else
                'mc_budget_capped' if name.endswith('-mccapped') else
                'mc_budget' if name.endswith('-mcbudget') else 'declared_budget',
                kv_floor=0, declared_budget_blocks=None if fixed else 32768,
                declared_budget_block_size=16)
            for key, value in expected_budget.items():
                if config.get(key) != value:
                    raise ValueError(f'{name}: {key} conflicts with the declared-budget caption')
        if name.endswith(('-mcstatic', '-mccapped', '-mcphase')):
            if (type(config.get('maximum_declared_blocks')) is not int or
                    config['maximum_declared_blocks'] != 39138 or
                    config.get('use_future_peak') is not name.endswith(('-mccapped', '-mcphase'))):
                raise ValueError(f'{name}: caption requires the shared39138 ceiling '
                                 'and its explicitly recorded future-peak setting')
        if mc_phase and config.get('allow_scheduled_prefill') is not name.endswith('-mcphase'):
            raise ValueError(f'{name}: phase caption requires its explicitly recorded '
                             'scheduled-prefill setting')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New SVG path.')
    parser.add_argument('--png', action='store_true', help='Also save a 180 dpi PNG.')
    parser.add_argument('--cell-root', type=Path,
                        help='Relocated raw cell directory for reading configs; saved analysis remains unchanged.')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.svg':
        parser.error('--output must name an SVG file.')
    outputs = [args.output] + ([args.output.with_suffix('.png')] if args.png else [])
    if any(path.exists() for path in outputs):
        parser.error('Output already exists; choose a new --output path.')
    source = json.loads(args.analysis.read_text())
    cells = {}
    for cell in source['cells']:
        name = Path(cell['cell']).name
        if name in cells:
            raise ValueError(f'Duplicate cell name: {name}')
        cells[name] = cell
    blocks, policies, progress = select_layout(cells)
    reservation = blocks == RESERVATION_BLOCKS
    headroom = blocks == HEADROOM_BLOCKS
    simple = blocks == SIMPLE_BLOCKS
    matched = blocks == DECLARED_MATCHED_BLOCKS
    declared = blocks in (DECLARED_BLOCKS, DECLARED_MATCHED_BLOCKS)
    mc = blocks == MC_BLOCKS
    mc_ablation = blocks == MC_ABLATION_BLOCKS
    mc_phase = blocks == MC_PHASE_BLOCKS
    fixed_cap = 177 if matched else 128
    names = [name for block in blocks for name in block]
    native_guard = (simple or declared or mc or mc_ablation or mc_phase) and native_guard_recheck(cells, names, args.cell_root)
    if declared and not native_guard:
        raise ValueError(f'Declared-budget comparison requires explicit native guards{fixed_cap}/256 and compiled max256.')
    if (mc or mc_ablation or mc_phase) and not native_guard:
        raise ValueError('MC-budget comparison requires explicit native guards256 and compiled max256 in every arm.')
    if len({cells[name]['workload_identity_sha256'] for name in names}) != 1:
        raise ValueError('All four arms must share the workload and external arrival trace.')
    cdfs = {(name, field): observed_cdf(cells[name], field)
            for name in names for field, _, _ in FIELDS}
    limits = {}
    for field, _, _ in FIELDS:
        values = [x for name in names for x, _ in cdfs[name, field][0]]
        # A zero gap is valid (e.g. a one-token output), so retain it on a
        # linear axis if present. Never hide zero observations on a log axis.
        log_axis = field == 'max_generation_gap_s' and values and min(values) > 0
        limits[field] = (min(values)*.8 if log_axis else 0.,
                         max(values)*1.08 if values and max(values) > 0 else 1.,
                         bool(log_axis))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'axes.linewidth': .8,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), sharex='col', sharey=True)
    fig.subplots_adjust(left=.09, right=.983, top=.825,
                        bottom=.35 if matched or mc_phase else .32 if native_guard else .29 if simple else .26,
                        wspace=.18, hspace=.43)
    fig.suptitle(('Strict decode vs. scheduled-prefill completion' if mc_phase else
                 'Static oversubscription vs. future-peak capped' if mc_ablation else
                 'Declared budget vs. guarded MC adaptation' if mc else
                 'Declared lifetime budget versus fixed concurrency' if declared else
                 'Native waiting guards 192 / 256; compiled max 256' if native_guard else
                 'Simple admission rules after lifecycle repair' if simple else
                 'Fixed 32-page headroom probe' if headroom else 'Known-prefill reservation probe' if reservation else
                 'Recovery progress release' if progress else 'One-shot admission probe')+
                 ': full-population latency CDFs',
                 y=.972, fontsize=15, fontweight='semibold')
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=2, label=label)
               for label, color, style in policies]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .936),
               ncol=2, frameon=False, columnspacing=3, handlelength=3.2)
    for row, block in enumerate(blocks):
        for col, (field, title, xlabel) in enumerate(FIELDS):
            ax = axes[row, col]
            xmin, xmax, log_axis = limits[field]
            for index, (name, (label, color, style)) in enumerate(zip(block, policies)):
                points, note = cdfs[name, field]
                if points:
                    xs = [xmin] + [x for x, _ in points] + [xmax]
                    ys = [0.] + [y for _, y in points] + [points[-1][1]]
                    ax.step(xs, ys, where='post', color=color, linestyle=style, linewidth=1.8)
                note_label = (('Strict decode' if index == 0 else 'Prefill completion allowed') if mc_phase else
                              ('Declared budget' if index == 0 else 'Guarded MC') if mc else
                              (f'Fixed{fixed_cap}' if index == 0 else 'Declared budget') if declared else label)
                ax.text(.97, .065+.085*(1-index), f'{note_label}: {note}',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                        color=color, bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=1))
            if log_axis:
                ax.set_xscale('log')
                xlabel += '; log scale'
            ax.set_title(f'{"abcdef"[row*3+col]}   {title}', loc='left', pad=10)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, 1.025)
            ax.set_xlabel(xlabel, labelpad=6)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.tick_params(axis='x', labelbottom=True)
            ax.grid(color='#E5E7EB', linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        axes[row, 0].set_ylabel(
            ('AB pair: 00 / 01' if row == 0 else 'BA pair: 03 / 02')+
            '\nFraction of all planned arrivals', labelpad=10)

    counts = ';  '.join(f'{name.removeprefix("probe-")}: '
                       f'{cells[name]["outcomes"]["completed"]}/{cells[name]["planned_requests"]} completed'
                       for name in (blocks[0][0], blocks[0][1], blocks[1][1], blocks[1][0]))
    caption = [counts,
        ('Execution order: 00 strict decode, 01 prefill completion, 02 prefill completion, 03 strict decode.' if mc_phase else
         'Execution order: 00 static, 01 future-peak capped, 02 future-peak capped, 03 static.' if mc_ablation else
         'Execution order: 00 declared budget, 01 guarded MC, 02 guarded MC, 03 declared budget.' if mc else
         f'Execution order: 00 fixed{fixed_cap}, 01 declared budget, 02 declared budget, 03 fixed{fixed_cap}.' if declared else
         'Execution order: 00 fixed192, 01 KV256, 02 KV256, 03 fixed192.' if simple else
         'Execution order: 00 baseline, 01 headroom32, 02 headroom32, 03 baseline.' if headroom else
         'Execution order: 00 baseline, 01 reservation, 02 reservation, 03 baseline.' if reservation else
         'Execution order: 00 timer, 01 progress, 02 progress, 03 timer.' if progress else
         'Execution order: 00 baseline, 01 delay, 02 delay, 03 baseline.')+
        ' Two run-level repeats per policy; no request-level CI.',
        'TTFT and completion time include waiting from external arrival. All planned requests remain in every denominator.',
        'Natural EOS; output amounts may differ. Maximum generation gaps use host-return token timestamps.',
        'Descriptive exploration on one workload; no independent confirmation or equal-work speedup claim.']
    if progress:
        caption.insert(2, 'Both arms hold 100 ms; progress then waits for all trigger-time recoveries to output or finish, '
                       'up to a 250 ms extra-gate cutoff.')
    if reservation:
        caption.insert(2, 'One target at the first known-prefill footprint excess; release when it fits, '
                       'reservations clear, or 250 ms expires. No minimum hold.')
    if headroom:
        caption.insert(2, 'Ordinary fixed headroom: first native-fit target with full-fit margin <32 pages; '
                       'release at margin >=32 or 250 ms. No minimum hold.')
    if simple:
        caption[2:2] = [
            'Rules: Fixed192 cap192 / KV floor0; KV256 cap256 / floor3277 (10% KV), bypassed at external request age10s.',
            'Common setup: release old/warmup raw; pre-arrival gc.collect(); passive GC in all arms. No extra admission probe.']
    if declared:
        caption[2:2] = [
            f'Rules: fixed cap{fixed_cap}; declared budget32768 pages / cap256. Both KV floor0; budget has no age override.',
            f'Native waiting guards{fixed_cap} / 256; compiled max256 in every arm. Charges use known prompt + declared output cap.',
            'Common setup: release old/warmup raw; pre-arrival gc.collect(); passive GC. Ordinary capacity control, not recovery novelty.']
    if mc:
        caption[2:2] = [
            'Rules: full declared sum vs. guarded future-round peak; both cap256, KV floor0, pool32768 pages; no age override.',
            'Existing MC-Benchmark principle, production-guarded adaptation: not a new algorithm. Virtual full charges may exceed the pool.',
            'Native guards256 / compiled max256; complete Gate retained. Common raw release, pre-arrival cleanup and passive GC.']
    if mc_ablation:
        caption[2:2] = [
            'Both arms: cap256, KV floor0, physical pool32768 pages, declared ceiling39138; no age override.',
            'Identical complete native guards256 / compiled max256. Static skips peak computation; capped requires future peak <=32768.',
            'Development-calibrated ceiling; ablation of an existing principle, not a new algorithm. Common cleanup and passive GC.']
    if mc_phase:
        caption[2:2] = [
            'Both: cap/native/compiled256, KV floor0, physical pool32768, declared ceiling39138, future peak enabled; no age override.',
            'Phase arm permits old prefill completing this step: computed + scheduled == num_tokens. Unfinished prefill remains rejected.',
            'The full prompt must finish within the current schedule. Running order / token quantum unchanged; no admission probes.',
            'Nearest-neighbor adaptation, not a new method. Shared development-calibrated ceiling, cleanup and passive GC.']
    if matched:
        caption.insert(3, 'Development mean-charge calibration: 177 = floor(32768/ceil(70702/384)); '
                       'not a globally optimal cap or independent confirmation.')
    if native_guard and simple:
        caption.insert(3, 'Native waiting guards: Fixed192=192, KV256=256; compiled max=256 in all arms. '
                       'Complete admission Gate retained.')
    positions = ((.261, .231, .201, .171, .141, .111, .081, .051, .021) if matched or mc_phase else
                 (.231, .201, .171, .141, .111, .081, .051, .021) if native_guard else
                 (.208, .177, .146, .115, .084, .053, .022) if simple else
                 (.194, .162, .130, .098, .066, .034) if progress or reservation or headroom else
                 (.194, .157, .12, .083, .046))
    for y, line in zip(positions, caption):
        fig.text(.09, y, line, fontsize=8.6, color='#555555', va='top')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        fig.savefig(handle, format='svg', metadata={'Date': None,
            'Description': ('Native waiting guards are 192 for Fixed192 and 256 for KV256; compiled engine '
                'maximum is 256 for every arm. Complete unique-unfinished admission Gate retained. '
                if native_guard and simple else '')+('Four strict-decode / scheduled-prefill-completion runs in AB and BA order. '
                'Both use future-peak checking, cap/native/compiled maximum256, KV floor0, physical budget32768 '
                'and the shared development-calibrated maximum full declared charge39138. Only the phase arm '
                'allows an old prefill whose computed + scheduled == num_tokens and whose full prompt '
                'finishes within the current schedule; genuinely unfinished prefill remains rejected. '
                'The complete admission guard, running order and token quantum remain unchanged. '
                'No admission probes or age override. Nearest-neighbor adaptation, not a new method. '
                if mc_phase else 'Four static-oversubscription / future-peak-capped runs in AB and BA order. '
                'Both use cap256, KV floor0, physical budget32768 pages and the same development-calibrated '
                'maximum full declared charge39138. Native waiting guards256 and compiled maximum256; '
                'complete admission and identical runtime guards retained. Static oversubscription does not '
                'compute a future peak; the capped arm additionally requires that peak to fit physical budget32768. '
                'No uncomputed static peak or new-algorithm claim. No age override. '
                if mc_ablation else 'Four ordinary declared-budget / guarded MC-budget runs in AB and BA order. '
                'Both use cap256, KV floor0 and physical budget32768 pages; native waiting guards256 and compiled maximum256. '
                'MC-Benchmark future-round FIFO/upper-length principle with production guards, not a new algorithm. '
                'Successful MC relaxations may make virtual full declared charges exceed the pool; physical peak remains guarded. '
                'All planned external arrivals remain CDF denominators; natural EOS output amounts may differ. '
                if mc else 'Four ordinary capacity-control runs in AB and BA order: '
                f'fixed cap{fixed_cap} versus declared lifetime budget32768 pages with cap256. Native waiting guards{fixed_cap}/256; '
                'compiled engine maximum256 in all arms. Both KV floor0; budget charges ceil((known prompt+'
                'declared maximum output)/16), with no age override. All planned arrivals remain CDF denominators. '
                'No recovery-specific novelty or independent-confirmation claim. ' if declared else
                'Four simple-rule runs after common measurement lifecycle repair, in AB and BA order. '
                'Fixed192 uses complete cap192 and KV floor0; KV256 uses cap256, floor3277 and a 10s '
                'external-age bypass. All arms release old/warmup raw, collect before external arrivals '
                'and passively observe GC; no extra admission probe. ' if simple else
                'Four real admission-probe runs: baseline/fixed 32-page headroom in AB and BA order. '
                'Ordinary headroom probe, not a recovery signal: release at 32-page full-fit margin or '
                '250 ms cutoff, no minimum hold. ' if headroom else
                'Four real admission-probe runs: baseline/reservation in AB and BA order. '
                'Only first conflict target is gated; known-footprint release or 250 ms cutoff, no minimum hold. '
                if reservation else 'Four real admission-probe runs: timer/progress in AB and BA order. '
                'Both hold 100 ms; progress waits for recovery output or finish, with a 250 ms extra-gate cutoff. '
                if progress else 'Four real admission-probe runs: baseline/delay in AB and BA order. ')+
                ('Fixed cap177 is development mean-charge calibration: '
                 '177 = floor(32768/ceil(70702/384)); not a globally optimal cap or independent confirmation. '
                 if matched else '')+
                'Full-population latency CDFs; natural EOS. Two run-level repeats per policy; '
                'no request-level confidence intervals or independent-confirmation claim.'})
    if args.png:
        with args.output.with_suffix('.png').open('xb') as handle:
            fig.savefig(handle, format='png', dpi=180)
    plt.close(fig)
    for path in outputs:
        print(path.resolve())


if __name__ == '__main__':
    main()
