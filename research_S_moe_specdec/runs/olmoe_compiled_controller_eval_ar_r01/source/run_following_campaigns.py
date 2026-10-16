"""Continue the two already specified S campaigns after the graph comparison.

This CPU coordinator never acquires the GPU lock itself. Each existing campaign
owns its lock, budget, source snapshot and child lifetime. A failed predecessor
or stage stops the sequence; rerunning cannot overwrite a run directory.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs/olmoe_following_campaigns_r01'
    output.mkdir(exist_ok=False)
    stages = [
        ('olmoe_ngram_compiled_r01', 'olmoe_ngram_compiled.json', []),
        ('olmoe_fullbudget_calibration_r01', 'olmoe_ngram_fullbudget_calibration.json',
         ['--mode', 'static', '--proposal-budget', '7']),
    ]
    state = {'status': 'waiting_for_predecessor', 'pid': os.getpid(),
             'started_unix_s': time.time(), 'predecessor': 'olmoe_kernel_graph_r01',
             'stages': stages, 'completed_stages': []}

    def save():
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(output / 'status.json')

    save()
    deadline = time.monotonic() + 21600
    while True:
        previous = json.loads((root / 'runs/olmoe_kernel_graph_r01/status.json').read_text())
        if previous['status'] == 'complete':
            break
        if previous['status'] not in ('waiting_for_lock', 'running'):
            state.update(status='predecessor_needs_attention', predecessor_status=previous['status'])
            save()
            return
        if time.monotonic() >= deadline:
            state['status'] = 'predecessor_wait_expired'
            save()
            return
        time.sleep(10)
    for run, config, extra in stages:
        command = [sys.executable, '-u', 'code/run_campaign.py', '--config', f'configs/{config}',
                   '--run', run, '--skip-trace', '--wait-lock-seconds', '21600', *extra]
        state.update(status='following_campaign', active_run=run, command=command)
        save()
        with (output / f'{run}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            code = child.wait()
        result_path = root / 'runs' / run / 'status.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if code != 0 or result.get('status') != 'complete':
            state.update(status='stage_needs_attention', returncode=code,
                         stage_status=result.get('status', 'missing'))
            save()
            return
        state['completed_stages'].append(run)
        save()
    state.update(status='complete', end_unix_s=time.time())
    save()


if __name__ == '__main__':
    main()
