#!/usr/bin/env python3
"""Frozen full-service plotting adapted to a single asynchronous-recovery start gate."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ACK_PLOT_SHA = '3e7d29e4b78b8103a1ae999c03050f2a4fb636deaa33d4f95511dd64e9954e22'


def adapted_namespace():
    path = BASE/'recovery_start_yield/plot_ack_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ACK_PLOT_SHA:
        raise RuntimeError('Frozen ACK figure adapter changed')
    spec = importlib.util.spec_from_file_location('start_gate_frozen_ack_plot', path)
    ack = importlib.util.module_from_spec(spec); spec.loader.exec_module(ack)
    namespace = ack.adapted_namespace(); replace_once = ack.replace_once
    namespace['__doc__'] = __doc__
    namespace['LABELS'] = ('N1 native', 'G1 wait_release', 'G2 wait_release', 'N2 native')
    # The frozen request-path helper retains its original module globals.
    original = namespace['request_path'].__globals__
    source = inspect.getsource(original['started_cells']).replace('yield_once', 'wait_release')
    exec(compile(source, '<start-gate-figure-arms>', 'exec'), namespace)
    helpers, request_path = namespace['HELPERS'], namespace['request_path']

    def local_paths(cells):
        paths = []
        for arm, cell in cells:
            action = cell.get('recovery_start_gate_actions', {})
            if action.get('status') != 'ANALYZED':
                paths.append(dict(arm=arm, role='observation', note='Action records unavailable or unverified')); continue
            rows = action.get('rows', [])
            if not rows:
                paths.append(dict(arm=arm, role='no action', note='No legal opportunity observed; zero executed gate')); continue
            for row in rows:
                evidence = dict(row.get('observed_target_evidence') or {})
                # These are this target's actual subsequent LOADs, not a second request role.
                evidence['pending_load_jobs_at_decision'] = [dict(job_id=job.get('job_id'), observation=job)
                    for job in evidence.get('load_jobs_after_decision') or []]
                actual = row.get('actual_gate_executed')
                adapted = dict(decision_s=row.get('decision_s'), actual_yield_executed=actual)
                item = request_path(arm, adapted, evidence, 'gated recovery' if actual else 'native recovery (shadow)')
                release = row.get('release_s'); decision = row.get('decision_s')
                item['release'] = release-decision if actual and all(helpers.finite(value) for value in (release, decision)) else None
                item['release_reason'] = row.get('release_reason') if actual else None
                paths.append(item)
        return paths

    namespace['local_paths'] = local_paths
    source = inspect.getsource(original['main'])
    source = source.replace('recovery_start_yield_actions', 'recovery_start_gate_actions')
    source = replace_once(source, "cell['mode'] == 'yield_once'", "cell['mode'] == 'wait_release'")
    source = replace_once(source, 'figsize=(13.6, 10.2+extra_height)', 'figsize=(16.6, 10.2+extra_height)')
    source = replace_once(source, 'fig.add_gridspec(3, 2,', 'fig.add_gridspec(3, 3,')
    source = replace_once(source,
        "        ('flow_s', '(a) Complete-request latency', 'External arrival → completion (s)'),",
        "        ('ttft_s', '(a) Time to first token', 'External arrival → first output (s)'),\n"
        "        ('flow_s', '(b) Complete-request latency', 'External arrival → completion (s)'),")
    source = replace_once(source, '(b) Per-request maximum generation gap', '(c) Maximum generation gap')
    source = replace_once(source, "ax.legend(loc='lower right', frameon=False, fontsize=8)",
                          "ax.legend(loc='upper left' if key == 'flow_s' else 'lower right', frameon=False, fontsize=8)")
    source = replace_once(source, "(path['wait'], path['allocation'], path['schedule'])",
                          "(path['wait'], path['allocation'], path['schedule'], path['release'])")
    source = replace_once(source, "        label = ('' if path['next_output_observed'] else '≥ ')",
        "        if HELPERS.finite(path['release']):\n"
        "            ax.scatter(path['release'], y, marker='x', color='#8C681F', s=35, zorder=7)\n"
        "        label = ('' if path['next_output_observed'] else '≥ ')")
    source = replace_once(source,
        "        if path['loads']: note += ' | LOAD jobs='+str(len(path['loads']))+' (decision batch '+str(path['current_batch_count'])+')'",
        "        if path['loads']: note += ' | subsequent LOAD jobs='+str(len(path['loads']))\n"
        "        if path['release_reason']: note += ' | '+str(path['release_reason'])")
    source = replace_once(source,
        '(c) Separate request paths: recompute head and each recorded pending-LOAD beneficiary',
        '(d) Each selected recovery request: LOAD, scheduler ACK, allocation, and next output')
    source = replace_once(source,
        "        Line2D([], [], ls='', marker='|', color='#6F537F', label='Resumed schedule plan')],",
        "        Line2D([], [], ls='', marker='|', color='#6F537F', label='Resumed schedule plan'),\n"
        "        Line2D([], [], ls='', marker='x', color='#8C681F', label='Actual extra gate released')],")
    source = replace_once(source, 'ncol=3, frameon=False, fontsize=8)', 'ncol=4, frameon=False, fontsize=8)')
    source = replace_once(source,
        'One-round recovery-start yield: full-service outcomes and actual execution',
        'Asynchronous recovery-start gate: full-service outcomes and actual execution')
    source = replace_once(source, '    notes = []',
        "    notes = ['At most one selected target / 16 executed breaks; cohort completion, entry limit, or native safety exits release the gate.']")
    source = replace_once(source, 'These outcome differences cannot be attributed to yield.',
                          'These outcome differences cannot be attributed to the gate.')
    source = replace_once(source,
        'A pending-LOAD beneficiary label records the policy’s candidate role, not a proven beneficiary or earlier arrival. The recompute head is a separate request.',
        'The local row follows only the selected recovery request. Cohort completion is a recorded release condition, not proof of increased free GPU capacity.')
    source = replace_once(source,
        'A one-round break does not fix the delay or guarantee next-round allocation. Hollow LOAD markers cover recorded pending job IDs, including earlier-ready jobs.',
        'Release does not guarantee allocation or output; gate lifetime is not counterfactual added latency or savings. Hollow markers show this target’s actual subsequent LOADs.')
    exec(compile(source, '<start-gate-full-service-figure>', 'exec'), namespace)
    return namespace


def main():
    adapted_namespace()['main']()


if __name__ == '__main__':
    main()
