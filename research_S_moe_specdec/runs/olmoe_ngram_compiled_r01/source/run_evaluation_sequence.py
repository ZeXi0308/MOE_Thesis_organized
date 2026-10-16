"""Run the already specified evaluation after the development campaigns.

This CPU coordinator does not hold a GPU lock. Each child run_campaign owns
the shared lock and checks the cumulative 20-hour budget. No failed stage is
automatically repeated, and the fixed run names prevent accidental overwrite.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def replace_argument(argv, flag, value):
    result = list(argv)
    if result.count(flag) != 1:
        raise ValueError(f'expected exactly one {flag}')
    result[result.index(flag) + 1] = value
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs/olmoe_evaluation_sequence_r01'
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / 'coordinator.py')
    state = {'status': 'waiting_for_predecessor', 'pid': os.getpid(),
             'started_unix_s': time.time(), 'stages': []}

    def save():
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(output / 'status.json')

    def execute(label, argv, status_file=None):
        stage = {'label': label, 'argv': argv, 'started_unix_s': time.time()}
        state['stages'].append(stage)
        state.update(status='running_stage', active_stage=label)
        save()
        with (output / f'{label}.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            stage['pid'] = child.pid
            save()
            code = child.wait()
        stage.update(returncode=code, end_unix_s=time.time())
        if status_file:
            path = Path(status_file)
            path = path if path.is_absolute() else root / path
            if not path.resolve().is_relative_to((root / 'runs').resolve()):
                raise ValueError('child status must remain under project runs/')
            stage['result_status'] = json.loads(path.read_text())['status'] if path.exists() else 'missing'
        save()
        if code or (status_file and stage['result_status'] != 'complete'):
            raise RuntimeError(f'{label}: exit {code}, state {stage.get("result_status")}')

    def campaign(run, config, extra=()):
        execute(run, [sys.executable, '-u', 'code/run_campaign.py',
                      '--config', f'configs/{config}', '--run', run,
                      '--wait-lock-seconds', '21600', *extra],
                f'runs/{run}/status.json')

    def verify_draft(run, source):
        execute(f'{run}_draft_check', [sys.executable, '-u',
                'code/check_native_draft_smoke.py', '--run', f'runs/{run}',
                '--draft-source', source])

    def evaluate_manifest(path):
        commands = json.loads(path.read_text())['commands']
        roles = {'ar', 'static', 'native_unrestricted', 'native_threshold', 'adaptive'}
        if (len(commands) != 5 or len({item['run'] for item in commands}) != 5
                or {item['role'] for item in commands} != roles):
            raise ValueError('expected five unique policy runs and roles')
        for item in commands:
            if item['argv'][:3] != [sys.executable, '-u', 'code/run_campaign.py']:
                raise ValueError('unexpected evaluation command prefix')
            if Path(item['cwd']).resolve() != root.resolve():
                raise ValueError('unexpected evaluation working directory')
        for item in commands:
            execute(item['run'], item['argv'], item['status_file'])
        return {item['role']: item['argv'] for item in commands}

    save()
    try:
        deadline = time.monotonic() + 21600
        predecessor_path = root / 'runs/olmoe_following_campaigns_r01/status.json'
        while True:
            previous = json.loads(predecessor_path.read_text())
            if previous['status'] == 'complete':
                break
            if previous['status'] not in ('waiting_for_predecessor', 'following_campaign'):
                raise RuntimeError('predecessor needs attention: ' + previous['status'])
            # A stale status file is not a live process. Never restart it here.
            os.kill(previous['pid'], 0)
            if time.monotonic() >= deadline:
                raise TimeoutError('predecessor observation deadline; inspect the original process')
            time.sleep(10)

        preparation = 'runs/controller_preparation_eager_r01'
        execute('prepare_controller_evaluation', [sys.executable, '-u',
            'code/prepare_controller_evaluation.py', '--calibration-run',
            'runs/olmoe_fullbudget_calibration_r01', '--direct-run',
            'runs/olmoe_ngram_matrix_r02', '--output', preparation])
        by_role = evaluate_manifest(root / preparation / 'evaluation_commands.json')

        # Natural stopping checks the three principal mechanisms on the same
        # held-out population, with its own shorter evaluation configuration.
        for role in ('ar', 'static', 'adaptive'):
            run = f'olmoe_controller_natural_{role}_r01'
            argv = replace_argument(by_role[role], '--config', 'configs/olmoe_controller_natural_check.json')
            argv = replace_argument(argv, '--run', run)
            if '--natural-stop' not in argv:
                argv.append('--natural-stop')
            execute(run, argv, f'runs/{run}/status.json')

        # The ordinary neural draft is fully downloaded and content-verified.
        campaign('olmoe_dense_draft_smoke_r01', 'olmoe_dense_draft_smoke.json', ['--skip-trace'])
        verify_draft('olmoe_dense_draft_smoke_r01', 'draft_model')
        campaign('olmoe_dense_draft_reduced_r01', 'olmoe_dense_draft_reduced.json')
        execute('qwen3_awq_assets', [sys.executable, '-u', 'code/check_qwen3_awq_ready.py',
                '--output', 'runs/qwen3_awq_asset_readiness_before_gpu/ready.json'])
        campaign('qwen3_awq_ngram_smoke_r01', 'qwen3_awq_ngram_smoke.json', ['--skip-trace'])
        campaign('qwen3_awq_eagle3_smoke_r01', 'qwen3_awq_eagle3_smoke.json', ['--skip-trace'])
        verify_draft('qwen3_awq_eagle3_smoke_r01', 'eagle3')
        campaign('qwen3_awq_ngram_reduced_r01', 'qwen3_awq_ngram_reduced.json')
        campaign('qwen3_awq_eagle3_reduced_r01', 'qwen3_awq_eagle3_reduced.json')
        campaign('olmoe_wave_cap16_r01', 'olmoe_wave_cap16_diagnostic.json', ['--skip-trace'])
        campaign('olmoe_backfill_r01', 'olmoe_backfill_diagnostic.json', ['--skip-trace'])
        campaign('olmoe_ngram_lcc_r01', 'olmoe_ngram_lcc.json')
        campaign('olmoe_ngram_multi_news_r01', 'olmoe_ngram_multi_news.json')
        execute('granite3b_assets', [sys.executable, '-u', 'code/download_granite_target.py', '--check-only'])
        campaign('granite3b_ngram_smoke_r01', 'granite3b_ngram_smoke.json', ['--skip-trace'])
        campaign('granite3b_neural_draft_smoke_r01', 'granite3b_neural_draft_smoke.json', ['--skip-trace'])
        verify_draft('granite3b_neural_draft_smoke_r01', 'draft_model')
        campaign('granite3b_ngram_reduced_r01', 'granite3b_ngram_reduced.json')
        campaign('granite3b_neural_draft_reduced_r01', 'granite3b_neural_draft_reduced.json')

        # Each additional domain gets its own full-budget costs and held-out
        # policies. Direct gamma-specific costs never stand in for truncation.
        for domain in ('olmoe_dense_draft', 'qwen3_awq_ngram', 'qwen3_awq_eagle3',
                       'granite3b_ngram', 'granite3b_neural_draft'):
            campaign(f'{domain}_reduced_calibration_r01',
                     f'{domain}_reduced_calibration.json',
                     ['--gammas', '0', '3', '7', '--mode', 'static',
                      '--proposal-budget', '7', '--skip-trace'])
            execute(f'{domain}_prepare_evaluation', [sys.executable, '-u',
                    'code/prepare_reduced_evaluation.py', '--domain', domain])
            evaluate_manifest(root / f'runs/{domain}_controller_preparation_r01/evaluation_commands.json')
        state.update(status='complete', end_unix_s=time.time())
        save()
    except BaseException as error:
        state.update(status='needs_attention', error=repr(error), end_unix_s=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
