#!/usr/bin/env python3
"""Preserve the calibrated AWQ checkpoint and its unique common-BF16 master.

Default is a local plan. The root owner starts the real copy explicitly::

  python3 code/sync_derived_checkpoints.py --execute

Uses the existing authenticated SSH socket; stores no credentials and never
starts GPU work. Waits for the calibration run's terminal status AND SHA256.json,
then copies both checkpoint directories with resumable rsync. No --delete or
--inplace is used. Interrupted transfer data and replaced local files are kept.
The source inventory is frozen locally, so a changed remote checkpoint cannot
silently replace an earlier backup under the same run name.

Every copied file is SHA256 checked, including tokenizer/config/recipe files and
checkpoint_manifest.json itself. Existing weight hashes identify complete model
shards; partial files without a usable manifest are hashed on the server. A
failed run or missing/invalid checkpoint manifest is VERIFIED_LOCAL_PARTIAL,
never a complete model. Remote checkpoint_path fields are preserved verbatim;
use an exporter's --checkpoint-dir override when reading the local copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from calibration_common_basis import content_revision
from export_bf16_fp8_artifact import sha256
from export_common_basis_sidecars import load_pair


# This script executes only stdlib reads on the server. No checkpoint payloads
# cross stdout; only paths, sizes, timestamps and hashes are returned.
REMOTE_READER = r'''
import hashlib,json,stat,sys,time
from pathlib import Path
q=json.loads(sys.argv[1]); root=Path(q['root']); run=root/'runs'/q['run_name']
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def child(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe relative path')
    target=root/p
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('symlink/escaped checkpoint path')
    return target
if not (run/'status.json').is_file() or not (run/'SHA256.json').is_file():
    print(json.dumps({'ready':False}));sys.exit(0)
status=json.loads((run/'status.json').read_text())
if status.get('state') not in ('COMPLETE','FAILED'):
    print(json.dumps({'ready':False,'state':status.get('state')}));sys.exit(0)
status_sha=digest(run/'status.json'); archived=json.loads((run/'SHA256.json').read_text())
if archived.get('status.json')!=status_sha:raise ValueError('terminal status/archive SHA differs')
command=status['command']
for flag,name in (('--output',q['names']['awq']),('--common-basis-bf16-output',q['names']['bf16'])):
    if command.count(flag)!=1:raise ValueError('run does not identify this calibration output pair')
    declared=Path(command[command.index(flag)+1])
    if (root/declared).resolve()!=(root/'checkpoints'/name).resolve():
        raise ValueError('run command output differs from requested checkpoint')
answer={'ready':True,'run_state':status['state'],'run_status_sha256':status_sha,
        'run_archive_sha256':digest(run/'SHA256.json'),'checkpoints':{}}
if q['mode']=='probe':print(json.dumps(answer));sys.exit(0)
for fmt,name in q['names'].items():
    directory=child(root/'checkpoints',name); files=[]; expected={}
    if directory.exists() and not directory.is_dir():raise ValueError('checkpoint is not a directory')
    manifest=directory/'checkpoint_manifest.json'
    if q['mode']=='inventory' and manifest.is_file():
        try:
            rows=json.loads(manifest.read_text())['identity']['weight_files']
            for row in rows:
                p=child(directory,row['path']); value=row['sha256']
                if (p.is_file() and p.stat().st_size==row['bytes'] and
                    isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)):
                    expected[row['path']]=value
        except (ValueError,KeyError,TypeError):expected={}
    for p in sorted(directory.rglob('*')) if directory.exists() else []:
        if p.is_symlink():raise ValueError('checkpoint symlinks are unsupported')
        if p.is_dir():continue
        st=p.stat()
        if not stat.S_ISREG(st.st_mode):raise ValueError('checkpoint contains a nonregular file')
        relative=str(p.relative_to(directory))
        if any(part in ('.rsync-partial','.rsync-backups') for part in p.relative_to(directory).parts):
            raise ValueError('remote checkpoint has reserved transfer paths')
        row={'path':relative,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
        if q['mode']=='inventory':
            row['sha256']=expected.get(relative) or digest(p)
            row['sha256_source']='checkpoint_manifest' if relative in expected else 'remote_file_bytes'
        after=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('source file changed during inventory')
        files.append(row)
    answer['checkpoints'][fmt]={'name':name,'directory_exists':directory.is_dir(),'files':files}
print(json.dumps(answer))
'''


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def signature(inventory):
    """Stable source identity, also comparable with a stat-only second read."""
    return {key: inventory[key] for key in ('run_state', 'run_status_sha256', 'run_archive_sha256')} | {
        'checkpoints': {fmt: {'name': item['name'], 'directory_exists': item['directory_exists'],
            'files': [{key: row[key] for key in ('path', 'bytes', 'mtime_ns')} for row in item['files']]}
            for fmt, item in inventory['checkpoints'].items()}}


def safe_child(root, relative):
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('unsafe inventory path')
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('local symlink/escaped checkpoint path')
    return path


def verify_copy(local_root, inventory, config):
    """Hash every copied byte and separately decide whether model manifests are complete."""
    result = {}
    for fmt, item in inventory['checkpoints'].items():
        directory = local_root / 'checkpoints' / item['name']
        expected = {row['path']: row for row in item['files']}
        actual = set()
        for p in directory.rglob('*') if directory.exists() else []:
            relative = p.relative_to(directory)
            if any(part in ('.rsync-partial', '.rsync-backups') for part in relative.parts):
                continue  # Retained transfer/backup data is not a model file.
            if p.is_symlink():
                raise ValueError(f'local symlink: {p}')
            if p.is_file():
                actual.add(str(relative))
        if actual != set(expected):
            raise ValueError(f'{fmt}: local model file list differs from the frozen source inventory')
        for name, row in expected.items():
            path = safe_child(directory, name)
            if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError(f'{fmt}: local size/SHA256 mismatch: {name}')
        summary = {'files_verified': len(expected), 'bytes_verified': sum(r['bytes'] for r in expected.values()),
                   'complete_checkpoint': False, 'checkpoint_manifest_sha256': None}
        try:
            if 'checkpoint_manifest.json' not in expected:
                raise ValueError('checkpoint manifest absent; preserve partial output only')
            summary['checkpoint_manifest_sha256'] = expected['checkpoint_manifest.json']['sha256']
            manifest = json.loads((directory / 'checkpoint_manifest.json').read_text())
            identity = manifest['identity']
            if (manifest.get('checkpoint_kind') != 'local_derived' or manifest['format'] != fmt or
                    manifest.get('base_checkpoint') != config['base_checkpoint'] or
                    manifest.get('base_checkpoint_revision') != config['base_revision'] or
                    manifest.get('exact_base_revision_known') is not True or
                    identity['revision'] != content_revision(identity) or
                    manifest['checkpoint_fingerprint'] != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()):
                raise ValueError('derived checkpoint identity does not match the fixed calibration source')
            if expected['config.json']['sha256'] != identity['config_sha256']:
                raise ValueError('model config SHA256 disagrees with checkpoint manifest')
            rows = identity['weight_files']
            names = [row['path'] for row in rows]
            if not rows or len(set(names)) != len(names):
                raise ValueError('empty or duplicate checkpoint weight inventory')
            for row in rows:
                if any(expected[row['path']][key] != row[key] for key in ('bytes', 'sha256')):
                    raise ValueError('weight file disagrees with checkpoint manifest')
            if set(names) != {name for name in expected if name.endswith('.safetensors')}:
                raise ValueError('checkpoint manifest does not cover every safetensors shard')
            summary.update(complete_checkpoint=True, checkpoint_fingerprint=manifest['checkpoint_fingerprint'],
                           checkpoint_revision=identity['revision'])
        except (ValueError, KeyError, TypeError) as error:
            summary['incomplete_reason'] = f'{type(error).__name__}: {error}'
        result[fmt] = summary
    if all(row['complete_checkpoint'] for row in result.values()):
        try:
            load_pair(local_root / 'checkpoints' / inventory['checkpoints']['bf16']['name'] / 'checkpoint_manifest.json',
                      local_root / 'checkpoints' / inventory['checkpoints']['awq']['name'] / 'checkpoint_manifest.json')
        except (ValueError, KeyError, TypeError) as error:
            for row in result.values():
                row.update(complete_checkpoint=False, incomplete_reason=f'common-master pair: {error}')
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=root / 'configs/olmoe_awq_common_r01.json')
    p.add_argument('--run-name', help='Defaults to the calibration config run_name')
    p.add_argument('--local-root', type=Path, default=root)
    p.add_argument('--remote-root', default='/root/autodl-tmp/research_C_format_sched')
    p.add_argument('--control-path', default='/tmp/c-format-ssh.sock')
    p.add_argument('--wait-seconds', type=int, default=7200)
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--copy-timeout', type=int, default=7200)
    p.add_argument('--reserve-gib', type=float, default=20)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    name = args.run_name or config['run_name']
    names = {}
    for fmt, key in (('awq', 'output'), ('bf16', 'common_basis_bf16_output')):
        path = Path(config[key])
        if len(path.parts) != 2 or path.parts[0] != 'checkpoints':
            p.error('config outputs must be direct checkpoint subdirectories')
        names[fmt] = path.name
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value) for value in (name, *names.values())):
        p.error('run/checkpoint names must be safe single directory names')
    if names['awq'] == names['bf16'] or args.wait_seconds < 1 or args.copy_timeout < 1 or not 1 <= args.poll_seconds <= 60 or args.reserve_gib < 0:
        p.error('distinct checkpoints, positive timeouts, 1..60 polling interval and nonnegative reserve required')
    args.local_root = args.local_root.resolve()
    plan = {'run': name, 'remote_root': args.remote_root, 'local_root': str(args.local_root),
            'checkpoints': names, 'reserve_gib': args.reserve_gib,
            'state': 'PLAN_NO_COPY', 'requires': 'terminal run status plus SHA256.json',
            'partial_output_policy': 'preserved and verified as partial, never a complete model',
            'source_manifest_paths_rewritten': False}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if shutil.which('rsync') is None:
        raise RuntimeError('rsync is required')
    receipt = args.local_root / 'runs' / (name + '.derived_checkpoints.local_sync.json')
    inventory_path = receipt.with_name(name + '.derived_checkpoints.SHA256.json')
    ssh = ['ssh', '-S', args.control_path, '-p', '50328', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           'root@connect.westb.seetacloud.com']
    def remote(mode):
        query = {'mode': mode, 'root': args.remote_root, 'run_name': name, 'names': names}
        command = shlex.join(['/root/miniconda3/bin/python', '-', json.dumps(query)])
        response = subprocess.run([*ssh, command], input=REMOTE_READER, text=True, capture_output=True,
                                  timeout=20 if mode == 'probe' else args.copy_timeout)
        response.check_returncode()
        return json.loads(response.stdout)
    status = {**plan, 'state': 'WAITING_FOR_TERMINAL_RUN', 'started_unix_s': time.time()}
    write_json(receipt, status)
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            current = remote('probe')
            if current['ready']: break
            if time.monotonic() >= deadline:
                raise TimeoutError('calibration run did not reach an archived terminal state before deadline')
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        status.update(state='READING_REMOTE_INVENTORY', remote_run_state=current['run_state'])
        write_json(receipt, status)
        inventory = remote('inventory')
        if not inventory['ready'] or signature(inventory) != signature(remote('stat')):
            raise RuntimeError('remote output changed while inventory was collected')
        if inventory_path.exists() and json.loads(inventory_path.read_text()) != inventory:
            raise RuntimeError('source inventory differs from an earlier copy attempt; old data preserved')
        write_json(inventory_path, inventory)
        required = sum(row['bytes'] for item in inventory['checkpoints'].values() for row in item['files'])
        free = shutil.disk_usage(args.local_root).free
        status.update(source_bytes=required, local_free_bytes_before_copy=free, inventory_sha256=sha256(inventory_path))
        if free < required + int(args.reserve_gib * 2**30):
            raise RuntimeError('insufficient local disk for a full transfer/backup allowance plus reserve')
        attempt = str(time.time_ns())
        for item in inventory['checkpoints'].values():
            if not item['directory_exists']: continue
            target = safe_child(args.local_root, 'checkpoints/' + item['name'])
            target.mkdir(parents=True, exist_ok=True)
            # Refuse existing symlinks before rsync could traverse them.
            if any(path.is_symlink() for path in target.rglob('*')):
                raise ValueError('local destination contains symlinks')
            status.update(state='COPYING', copying=item['name']); write_json(receipt, status)
            command = ['rsync', '-a', '--checksum', '--partial', '--partial-dir=.rsync-partial',
                '--backup', '--backup-dir=' + str(target / '.rsync-backups' / attempt),
                '-e', shlex.join(ssh[:-1]),
                ssh[-1] + ':' + shlex.quote(args.remote_root + '/checkpoints/' + item['name'] + '/'), str(target) + '/']
            subprocess.run(command, check=True, timeout=args.copy_timeout)
        if signature(inventory) != signature(remote('stat')):
            raise RuntimeError('remote output changed during transfer; copied bytes retained, not accepted')
        status.update(state='VERIFYING_ALL_LOCAL_FILES'); write_json(receipt, status)
        status['checkpoints_verified'] = verify_copy(args.local_root, inventory, config)
        complete = all(item['complete_checkpoint'] for item in status['checkpoints_verified'].values())
        if inventory['run_state'] == 'COMPLETE' and not complete:
            raise ValueError('successful calibration run has incomplete/invalid derived model manifests')
        status.update(state='VERIFIED_LOCAL_COMPLETE' if complete and inventory['run_state'] == 'COMPLETE' else 'VERIFIED_LOCAL_PARTIAL',
                      completed_unix_s=time.time(), source_stable_during_copy=True,
                      usable_complete_pair=complete and inventory['run_state'] == 'COMPLETE')
        write_json(receipt, status)
        print(json.dumps(status, indent=2))
    except BaseException as error:
        status.update(state='COPY_OR_VERIFICATION_FAILED', ended_unix_s=time.time(),
                      error=f'{type(error).__name__}: {error}', usable_complete_pair=False,
                      retained_partial_files=True)
        write_json(receipt, status)
        raise


if __name__ == '__main__':
    main()
