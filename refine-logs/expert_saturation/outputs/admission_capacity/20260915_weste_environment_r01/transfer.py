"""Copy retained runtime/model directly between authorized hosts; no GPU work."""
import base64
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, '/root/autodl-tmp/moe-transfer-tools-20260915')
import paramiko

ROOT = Path('/root/autodl-tmp/moe-transfer-weste-26862-20260915-r01')
ITEMS = ['expert-saturation/vllm-0.26',
         'hf-cache/hub/models--allenai--OLMoE-1B-7B-0924']
FINGERPRINT = 'liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc'


class PinnedHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        if actual != FINGERPRINT:
            raise RuntimeError('Destination SSH host key does not match observed key')


def save(state):
    state['updated_unix_s'] = time.time()
    temp = ROOT / 'state.json.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(ROOT / 'state.json')


def run(password):
    state = dict(status='CONNECTING', pid=os.getpid(), started_unix_s=time.time(),
                 destination='connect.weste.seetacloud.com:26862',
                 source_base='/root/autodl-tmp', items=ITEMS, transferred_bytes=0)
    save(state)
    client, source = paramiko.SSHClient(), None
    try:
        client.set_missing_host_key_policy(PinnedHost())
        client.connect('connect.weste.seetacloud.com', port=26862, username='root',
                       password=password, look_for_keys=False, allow_agent=False,
                       timeout=30, auth_timeout=30, banner_timeout=30)
        password = None
        transport = client.get_transport()
        transport.set_keepalive(15)
        preflight = """from pathlib import Path
import json, shutil
root=Path('/root/autodl-tmp')
items=['expert-saturation/vllm-0.26','hf-cache/hub/models--allenai--OLMoE-1B-7B-0924']
assert all(not (root/p).exists() for p in items), 'Destination already exists; refusing overwrite'
assert shutil.disk_usage(root).free > 25000000000, 'Insufficient destination space'
receipt=root/'moe-transfer-weste-26862-20260915-r01'
receipt.mkdir(exist_ok=False)
(receipt/'source.json').write_text(json.dumps({'source':'westc:53036','items':items}))
print('DESTINATION_EMPTY_AND_RESERVED')
"""
        stdin, stdout, stderr = client.exec_command('/root/miniconda3/bin/python -')
        stdin.write(preflight)
        stdin.channel.shutdown_write()
        output, error = stdout.read().decode(), stderr.read().decode()
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError('Destination preflight failed: ' + error)
        state['destination_preflight'] = output.strip()
        channel = transport.open_session(window_size=64 * 1024 * 1024)
        channel.settimeout(180)
        channel.exec_command('tar -C /root/autodl-tmp -xf -')
        with (ROOT / 'source-tar.stderr').open('wb') as log:
            source = subprocess.Popen(['tar', '-C', '/root/autodl-tmp', '-cf', '-', *ITEMS],
                                      stdout=subprocess.PIPE, stderr=log)
            state.update(status='COPYING', source_tar_pid=source.pid)
            save(state)
            last = time.monotonic()
            while True:
                chunk = source.stdout.read(4 * 1024 * 1024)
                if not chunk:
                    break
                channel.sendall(chunk)
                state['transferred_bytes'] += len(chunk)
                if time.monotonic() - last >= 5:
                    save(state)
                    last = time.monotonic()
            source.stdout.close()
            state['source_returncode'] = source.wait()
        channel.shutdown_write()
        state['destination_stdout'] = channel.makefile('rb').read().decode()
        state['destination_stderr'] = channel.makefile_stderr('rb').read().decode()
        state['destination_returncode'] = channel.recv_exit_status()
        if state['source_returncode'] != 0 or state['destination_returncode'] != 0:
            raise RuntimeError('Source or destination tar failed; retain partial destination')
        state['status'] = 'COPIED_AWAITING_VERIFICATION'
    except BaseException as exc:
        state.update(status='FAILED_RETAIN_PARTIAL', error=repr(exc))
        if source is not None and source.poll() is None:
            source.terminate()  # Only the tar process created by this copy attempt.
            source.wait()
    finally:
        state['finished_unix_s'] = time.time()
        save(state)
        client.close()


if __name__ == '__main__':
    ROOT.mkdir(exist_ok=False)
    password = getpass.getpass('Destination password: ')
    pid = os.fork()
    if pid:
        print(json.dumps({'detached_copy_pid': pid, 'receipt': str(ROOT / 'state.json')}), flush=True)
        sys.exit(0)
    os.setsid()
    with open(os.devnull, 'rb') as src, (ROOT / 'worker.log').open('ab', buffering=0) as log:
        os.dup2(src.fileno(), 0)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    run(password)
    os._exit(0)
