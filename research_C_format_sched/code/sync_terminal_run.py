#!/usr/bin/env python3
"""Wait locally for one terminal GPU run, then pull and verify its raw files.

Uses an already authenticated SSH control socket. Stores no credentials, starts
no GPU work, and never deletes local or remote results. A SHA256 manifest is
required, so a status update before run_locked finishes archiving is not enough.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--wait-seconds', type=int, default=7200)
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--control-path', default='/tmp/c-format-ssh.sock')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', a.name):
        p.error('name must be a single run name')
    if a.wait_seconds < 1 or not 1 <= a.poll_seconds <= 60:
        p.error('positive wait and 1..60 second polling interval required')
    root = Path(__file__).resolve().parents[1]
    remote = '/root/autodl-tmp/research_C_format_sched/runs/' + a.name
    target = root / 'runs' / a.name
    receipt = root / 'runs' / (a.name + '.local_sync.json')
    ssh = ['ssh', '-S', a.control_path, '-p', '50328', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=10', 'root@connect.westb.seetacloud.com']
    status = {'run': a.name, 'state': 'WAITING_FOR_TERMINAL_MANIFEST',
              'started_unix_s': time.time(), 'remote': remote, 'local': str(target)}
    def save():
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(receipt)
    deadline = time.monotonic() + a.wait_seconds
    save()
    while time.monotonic() < deadline:
        try:
            command = 'test -f ' + shlex.quote(remote + '/SHA256.json') + ' && cat ' + shlex.quote(remote + '/status.json')
            result = subprocess.run([*ssh, command], text=True, capture_output=True, timeout=20)
            if result.returncode:
                status.update(last_check_unix_s=time.time(), last_check_exit=result.returncode)
                save()
                time.sleep(min(a.poll_seconds, max(0, deadline - time.monotonic())))
                continue
            terminal = json.loads(result.stdout)
            if terminal['state'] not in ('COMPLETE', 'FAILED'):
                raise RuntimeError('remote manifest exists for a nonterminal run')
            status.update(state='COPYING', remote_state=terminal['state'])
            save()
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', '-e', shlex.join(ssh[:-1]),
                'root@connect.westb.seetacloud.com:' + remote + '/', str(target) + '/'],
                check=True, timeout=1800)
            entries = json.loads((target / 'SHA256.json').read_text())
            total = 0
            for relative, expected in entries.items():
                path = (target / relative).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise ValueError('unsafe manifest path')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b''):
                        digest.update(chunk)
                        total += len(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError('local SHA256 mismatch: ' + relative)
            subprocess.run([sys.executable, str(root / 'code/verify_local_data.py')],
                           check=True, stdout=subprocess.DEVNULL)
            status.update(state='VERIFIED_LOCAL', files=len(entries), bytes=total,
                          completed_unix_s=time.time())
            save()
            print(json.dumps(status), flush=True)
            return
        except (OSError, subprocess.SubprocessError) as error:
            status.update(state='RETRYING_COPY_OR_CONNECTION', last_error=f'{type(error).__name__}: {error}')
            save()
            time.sleep(min(a.poll_seconds, max(0, deadline - time.monotonic())))
        except (ValueError, KeyError, RuntimeError) as error:
            status.update(state='VERIFICATION_FAILED', ended_unix_s=time.time(),
                          last_error=f'{type(error).__name__}: {error}')
            save()
            raise
    status.update(state='WAIT_TIMEOUT', ended_unix_s=time.time())
    save()
    raise SystemExit('No verified local copy before the bounded wait expired')


if __name__ == '__main__':
    main()
