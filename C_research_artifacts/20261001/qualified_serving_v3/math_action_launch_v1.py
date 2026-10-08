#!/usr/bin/env python3
"""Bounded exploratory native math pair; existing shared-lock/model lifecycle."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import qwen7b_launch_v2 as model

BASE = Path(__file__).resolve().parent
resource = model.resource
require, sha, save = resource.require, resource.sha, resource.save
INPUTS = BASE / 'math_inputs128_v1.json'
PROTOCOL = BASE / 'math_action_protocol_v1.json'


def unit(name, seqs, budget):
    out = BASE / name
    require(not out.exists() and not out.is_symlink(), 'existing immutable output')
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        model.lock_identity(fd)
        deadline = time.monotonic() + 1200
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'shared GPU lock wait expired')
                print(json.dumps(dict(status='WAITING_SHARED_LOCK', unit=name,
                                      launcher_pid=os.getpid(), time=time.time())), flush=True)
                time.sleep(15)
        out.mkdir(); resource.OUT = out
        record = dict(status='RUNNING', unit=name, launcher_pid=os.getpid(),
            start_unix_s=time.time(), lock_inode=resource.LOCK_ID,
            model_id=model.MODEL, revision=model.REVISION,
            inputs_sha256=sha(INPUTS), protocol_sha256=sha(PROTOCOL),
            cell_sha256=sha(BASE / 'math_action_cell_v1.py'),
            launcher_sha256=sha(__file__), max_num_seqs=seqs,
            max_num_batched_tokens=budget, scope='Exploratory paired full-policy runs; not held-out confirmation.')
        try:
            record['gpu_before'] = resource.gpu_idle()
            save(out / 'launcher-receipt.json', record)
            require(shutil.disk_usage(BASE).free >= model.GIB, 'root disk reserve below 1 GiB')
            manifest = model.frozen_manifest()
            identity = model.stage_identity(manifest)
            files = model.inventory(manifest, complete=True)
            save(out / 'model-reverified.json', dict(status='VERIFIED', files=files, **identity))
            require(model.memory_headroom() >= 24 * model.GIB, 'RAM headroom below 24 GiB')
            resource.gpu_idle()
            resource.bounded([sys.executable, str(BASE / 'math_action_cell_v1.py'),
                '--model-dir', str(model.STAGE), '--inputs', str(INPUTS),
                '--output', str(out / 'native'), '--expected-gpu-uuid', resource.GPU,
                '--kv-bytes', '8589934592', '--max-seqs', str(seqs),
                '--batch-tokens', str(budget), '--max-model-len', '4096'],
                'native.log', 1200, fd)
            for filename, key, expected in [('status.json', 'status', 'COMPLETE'),
                    ('native-drain.json', 'status', 'QUALIFIED'),
                    ('measured-pressure.json', 'status', 'COMPLETE')]:
                require(json.loads((out / 'native' / filename).read_text())[key] == expected,
                        'incomplete math cell: ' + filename)
            record.update(status='COMPLETE', gpu_after=resource.gpu_idle())
        except BaseException as exc:
            record.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['end_unix_s'] = time.time()
            save(out / 'launcher-receipt.json', record)
    finally:
        os.close(fd)


def main():
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    model.install_signals()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['pair', 'native128-4096'], default='pair')
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    for filename, digest in protocol['execution_sha256'].items():
        require(sha(BASE / filename) == digest, 'execution file differs: ' + filename)
    if args.arm == 'pair':
        for name, seqs, budget in [('qwen7b-math-native128-v1', 128, 2048),
                                   ('qwen7b-math-fixed64-v1', 64, 2048)]:
            unit(name, seqs, budget)
    else:
        unit('qwen7b-math-native128-b4096-v1', 128, 4096)


if __name__ == '__main__':
    main()
