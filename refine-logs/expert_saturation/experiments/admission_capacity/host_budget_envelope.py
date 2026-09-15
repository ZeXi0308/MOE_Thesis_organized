"""Owned cgroup-v2 campaign envelope; the monitor stays outside the budget.

Only kernel-accounted host memory is limited. RSS is a separate sampled sum,
which may double-count shared pages. This module does not import the runtime.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


BOOTSTRAP = """import json,os,sys
from pathlib import Path
p=Path(sys.argv[1])/'cgroup.procs'
p.write_text(str(os.getpid()))
if str(os.getpid()) not in p.read_text().split():
    raise RuntimeError('cgroup membership readback failed; command not executed')
receipt=Path(sys.argv[2]); temp=receipt.with_suffix('.tmp')
temp.write_text(json.dumps(dict(pid=os.getpid(),membership_verified=True,exec_attempted=True)))
temp.replace(receipt)
os.execvpe(sys.argv[3],sys.argv[3:],os.environ)
"""
FIELDS = ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmHWM', 'VmLck', 'VmPin', 'VmSwap')


def read(path):
    return path.read_text().strip()


def populated(events):
    return int(dict(line.split() for line in events.splitlines())['populated']) != 0


def is_cgroup2(parent):
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), left.split()[4])
        if right.split()[0] == 'cgroup2' and parent.is_relative_to(Path(mount)):
            return True
    return False


def create_owned_group(parent, budget, swap, receipt):
    if budget <= 0 or swap < 0:
        raise ValueError('explicit positive budget and nonnegative swap bytes required')
    parent = parent.resolve(strict=True)
    if not is_cgroup2(parent) or 'memory' not in read(parent/'cgroup.subtree_control').split():
        raise RuntimeError('writable delegated cgroup-v2 memory controller required')
    group = parent/('moe-host-' + uuid.uuid4().hex)
    group.mkdir()  # Exclusive ownership; never reuse or mutate the parent.
    receipt['owned_cgroup'] = str(group)
    for name, value in [('memory.max', budget), ('memory.swap.max', swap)]:
        (group/name).write_text(str(value))
        actual = read(group/name)
        receipt.setdefault('limit_readback', {})[name] = actual
        if actual != str(value):
            raise RuntimeError(f'{name} readback mismatch: requested {value}, got {actual}')
    if read(group/'cgroup.procs'):
        raise RuntimeError('new owned cgroup unexpectedly populated')
    return group


def process_snapshot(pid, proc_root=Path('/proc')):
    stat = read(proc_root/str(pid)/'stat').rsplit(')', 1)[1].split()
    status = read(proc_root/str(pid)/'status')
    fields = {}
    for key in FIELDS:
        match = re.search(r'^' + key + r':\s+(\d+)\s+kB$', status, re.M)
        fields[key + '_bytes'] = int(match[1])*1024 if match else None
    return dict(pid=pid, starttime_ticks=int(stat[19]), **fields)


def snapshot(group):
    wall, cpu = time.monotonic(), time.process_time()
    row = dict(unix_s=time.time(), monotonic_s=wall, members=[], errors=[])
    for name in ('memory.current', 'memory.peak', 'memory.events', 'memory.stat',
                 'memory.swap.current', 'memory.swap.max', 'memory.max', 'cgroup.events'):
        try:
            row[name] = read(group/name)
        except OSError as exc:
            row['errors'].append(dict(path=name, error=str(exc)))
    pids = set()
    for path in group.rglob('cgroup.procs'):
        try:
            pids.update(int(v) for v in read(path).split())
        except (OSError, ValueError) as exc:
            row['errors'].append(dict(path=str(path), error=str(exc)))
    for pid in sorted(pids):
        try:
            row['members'].append(process_snapshot(pid))
        except (OSError, ValueError, IndexError) as exc:
            row['errors'].append(dict(pid=pid, error=str(exc)))
    row['sampled_rss_sum_bytes'] = sum(r['VmRSS_bytes'] or 0 for r in row['members'])
    try:
        row['monitor_self'] = process_snapshot(os.getpid())
    except (OSError, ValueError, IndexError) as exc:
        row['errors'].append(dict(monitor_error=str(exc)))
    row.update(sample_wall_s=time.monotonic()-wall, sample_cpu_s=time.process_time()-cpu)
    return row


def save_receipt(out, receipt):
    temp = out/'terminal.json.tmp'
    temp.write_text(json.dumps(receipt, indent=2)+'\n')
    temp.replace(out/'terminal.json')


def run_budgeted(command, *, parent, budget_bytes, swap_bytes, output, timeout_s=600):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='NOT_ENFORCED', command=command, launcher_started=False,
                   budget_bytes=budget_bytes, swap_bytes=swap_bytes, started_unix_s=time.time(),
                   sample_period_s=1.0, monitor_pid=os.getpid(), memory_scope='owned cgroup charge',
                   rss_scope='sampled member RSS sum; shared pages may be counted repeatedly')
    group = proc = None
    save_receipt(out, receipt)
    try:
        if not command or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('command and positive timeout required')
        group = create_owned_group(Path(parent), budget_bytes, swap_bytes, receipt)
        receipt.update(status='ENFORCED_STARTING', enforced=True)
        save_receipt(out, receipt)
        with (out/'process.log').open('x') as log, (out/'samples.jsonl').open('x') as samples:
            proc = subprocess.Popen([sys.executable, '-c', BOOTSTRAP, str(group), str(out/'membership.json'), *command],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(launcher_pid=proc.pid, launcher_started=True, status='RUNNING')
            save_receipt(out, receipt)
            deadline = time.monotonic()+timeout_s
            while True:
                started = time.monotonic()
                row = snapshot(group)
                samples.write(json.dumps(row, separators=(',', ':'))+'\n'); samples.flush()
                receipt['last_sample'] = row
                if proc.poll() is not None and (proc.returncode != 0 or not populated(row['cgroup.events'])):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('owned campaign exceeded timeout')
                time.sleep(min(max(0, 1.0-(time.monotonic()-started)), max(0, deadline-time.monotonic())))
            receipt.update(status='COMPLETE' if proc.returncode == 0 else 'COMMAND_FAILED',
                           returncode=proc.returncode)
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        if receipt.get('launcher_started'):
            receipt['status'] = 'TIMED_OUT' if isinstance(exc, TimeoutError) else 'FAILED'
        else:
            receipt['status'] = 'NOT_ENFORCED' if not receipt.get('enforced') else 'START_FAILED'
    finally:
        owned = group or (Path(receipt['owned_cgroup']) if 'owned_cgroup' in receipt else None)
        if (out/'membership.json').exists():
            try:
                receipt['bootstrap_membership'] = json.loads((out/'membership.json').read_text())
            except (OSError, ValueError) as exc:
                receipt['membership_read_error'] = str(exc)
        if proc is not None and proc.poll() is None:
            proc.kill()  # This Popen child only, including a failed bootstrap.
            proc.wait()
        if owned is not None:
            try:
                if populated(read(owned/'cgroup.events')):
                    (owned/'cgroup.kill').write_text('1')  # Only the exclusively created group.
                receipt['final_sample'] = snapshot(owned)
                owned.rmdir()  # Never recursively remove a filesystem tree.
            except (OSError, ValueError, KeyError) as exc:
                receipt['cleanup_error'] = str(exc)
        receipt['finished_unix_s'] = time.time()
        save_receipt(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--budget-bytes', required=True, type=int)
    parser.add_argument('--swap-bytes', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    result = run_budgeted(command, parent=args.parent, budget_bytes=args.budget_bytes,
                          swap_bytes=args.swap_bytes, output=args.output, timeout_s=args.timeout_s)
    print(json.dumps({'status': result['status'], 'receipt': str(args.output/'terminal.json')}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    sys.exit(main())
