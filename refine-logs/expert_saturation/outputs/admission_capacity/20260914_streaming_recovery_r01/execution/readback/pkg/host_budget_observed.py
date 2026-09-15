"""Launch one command under an existing shared cgroup limit and observe it read-only."""
import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path

try:
    from host_budget_envelope import process_snapshot
except ImportError:  # Standalone preparation worktree; main already owns this helper.
    _FIELDS = ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmHWM', 'VmLck', 'VmPin', 'VmSwap')

    def process_snapshot(pid, proc_root=Path('/proc')):
        stat = (proc_root/str(pid)/'stat').read_text().strip().rsplit(')', 1)[1].split()
        status = (proc_root/str(pid)/'status').read_text().strip()
        values = {}
        for key in _FIELDS:
            match = re.search(r'^'+key+r':\s+(\d+)\s+kB$', status, re.M)
            values[key+'_bytes'] = int(match[1])*1024 if match else None
        return dict(pid=pid, starttime_ticks=int(stat[19]), **values)


CGROUP_FIELDS = ('memory.max', 'memory.current', 'memory.peak', 'memory.events',
                 'memory.swap.max', 'memory.swap.current', 'memory.swap.events')
SCOPE = {
    'evidence_type': 'OBSERVED_ONLY',
    'parent_memory_scope': 'Shared parent cgroup charge/peak/events; parent peak is not this command peak.',
    'rss_scope': 'Sampled root-plus-PPid-descendant RSS sum; shared pages may be counted repeatedly.',
    'discovery_scope': 'About 1 Hz PPid scans may miss short-lived or reparented descendants.',
    'enforcement_scope': 'No cgroup or rlimit is created or changed; this is not an independent process-tree hard cap.'}


def parent_snapshot(parent):
    row = {'path': str(parent), 'values': {}, 'errors': []}
    for name in CGROUP_FIELDS:
        try:
            row['values'][name] = (parent/name).read_text().strip()
        except OSError as exc:
            row['errors'].append({'path': str(parent/name), 'error': str(exc)})
    return row


def process_tree_snapshot(root_pid, root_starttime, proc_root=Path('/proc')):
    meta, errors = {}, []
    if root_starttime is None:
        return [], [{'pid': root_pid, 'error': 'root starttime was not captured'}], 0
    try:
        paths = list(proc_root.iterdir())
    except OSError as exc:
        return [], [{'path': str(proc_root), 'error': str(exc)}], 0
    for path in paths:
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().strip().rsplit(')', 1)[1].split()
            meta[int(path.name)] = (int(fields[1]), int(fields[19]))
        except (OSError, ValueError, IndexError) as exc:
            errors.append({'pid': int(path.name), 'path': str(path/'stat'), 'error': str(exc)})
    if root_pid not in meta or (root_starttime is not None and meta[root_pid][1] != root_starttime):
        errors.append({'pid': root_pid, 'error': 'root missing or PID starttime changed'})
        return [], errors, 0
    scoped = {root_pid}
    while True:
        added = {pid for pid, (ppid, _) in meta.items() if ppid in scoped} - scoped
        if not added:
            break
        scoped.update(added)
    members = []
    for pid in sorted(scoped):
        try:
            row = process_snapshot(pid, proc_root)
            if row['starttime_ticks'] != meta[pid][1]:
                raise ValueError('PID starttime changed during sample')
            members.append(dict(row, ppid=meta[pid][0]))
        except (OSError, ValueError, IndexError) as exc:
            errors.append({'pid': pid, 'error': str(exc)})
    return members, errors, sum(row.get('VmRSS_bytes') or 0 for row in members)


def sample(parent, root_pid, root_starttime, proc_root=Path('/proc')):
    wall, cpu = time.monotonic(), time.process_time()
    members, errors, rss = process_tree_snapshot(root_pid, root_starttime, proc_root)
    row = {'unix_s': time.time(), 'monotonic_s': wall, 'parent_cgroup': parent_snapshot(parent),
           'root_pid': root_pid, 'root_starttime_ticks': root_starttime, 'members': members,
           'process_errors': errors, 'sampled_process_tree_rss_sum_bytes': rss}
    row.update(sampling_wall_s=time.monotonic()-wall, sampling_cpu_s=time.process_time()-cpu)
    return row


def save(out, receipt):
    temp = out/'terminal.json.tmp'
    temp.write_text(json.dumps(receipt, indent=2)+'\n')
    temp.replace(out/'terminal.json')


def run_observed(command, *, parent, expected_parent_memory_max, output,
                 declared_host_kv_offload_bytes, interval_s=1.0, proc_root=Path('/proc')):
    if (not command or expected_parent_memory_max <= 0 or declared_host_kv_offload_bytes < 0
            or not math.isfinite(interval_s) or interval_s <= 0):
        raise ValueError('command, positive expected parent limit/interval, and nonnegative offload are required')
    parent, out = Path(parent), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='OBSERVED_ONLY_PREFLIGHT', command=list(command), launched=False,
        expected_parent_memory_max=str(expected_parent_memory_max),
        declared_host_kv_offload_bytes=declared_host_kv_offload_bytes,
        declared_host_kv_offload_semantics='caller declaration; monitor does not infer runtime configuration',
        interval_s=interval_s, monitor_pid=os.getpid(), started_unix_s=time.time(), scope=SCOPE)
    receipt['prelaunch_parent'] = parent_snapshot(parent)
    try:
        receipt['monitor_cgroup_membership'] = (proc_root/'self/cgroup').read_text().strip()
    except OSError as exc:
        receipt['monitor_cgroup_read_error'] = str(exc)
    actual = receipt['prelaunch_parent']['values'].get('memory.max')
    if actual != str(expected_parent_memory_max):
        receipt.update(status='REFUSED_PARENT_LIMIT_MISMATCH', observed_parent_memory_max=actual,
                       finished_unix_s=time.time())
        save(out, receipt)
        return receipt
    proc = None
    try:
        with (out/'process.log').open('x') as log, (out/'samples.jsonl').open('x') as samples:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            receipt.update(status='RUNNING_OBSERVED_ONLY', launched=True, root_pid=proc.pid)
            try:
                receipt['root_starttime_ticks'] = process_snapshot(proc.pid, proc_root)['starttime_ticks']
            except (OSError, ValueError, IndexError) as exc:
                receipt.update(root_starttime_ticks=None, root_identity_error=str(exc))
            save(out, receipt)
            total_wall = total_cpu = 0.0
            sample_count = 0
            while True:
                started = time.monotonic()
                row = sample(parent, proc.pid, receipt['root_starttime_ticks'], proc_root)
                samples.write(json.dumps(row, separators=(',', ':'))+'\n')
                samples.flush()
                sample_count += 1
                total_wall += row['sampling_wall_s']
                total_cpu += row['sampling_cpu_s']
                if proc.poll() is not None:
                    break
                delay = max(0.0, interval_s-(time.monotonic()-started))
                try:
                    proc.wait(timeout=delay)
                except subprocess.TimeoutExpired:
                    pass
            receipt.update(returncode=proc.returncode, samples_written=sample_count,
                           sampling_wall_s=total_wall, sampling_cpu_s=total_cpu,
                           status='COMPLETE_OBSERVED_ONLY' if proc.returncode == 0 else 'CHILD_FAILED_OBSERVED_ONLY')
    except BaseException as exc:
        receipt.update(status='LAUNCH_FAILED_OBSERVED_ONLY' if proc is None else 'MONITOR_FAILED_OBSERVED_ONLY',
                       error=f'{type(exc).__name__}: {exc}')
        if proc is not None and proc.poll() is None:
            receipt['status'] = 'MONITOR_FAILED_WAITING_FOR_CHILD'
            save(out, receipt)  # Keep the original monitor error before blocking.
            while proc.poll() is None:
                try:
                    proc.wait()
                except BaseException as wait_exc:
                    receipt.setdefault('child_wait_errors', []).append(f'{type(wait_exc).__name__}: {wait_exc}')
                    save(out, receipt)
            receipt.update(status='MONITOR_FAILED_CHILD_TERMINAL', returncode=proc.returncode)
    finally:
        receipt['child_running_at_monitor_exit'] = proc is not None and proc.poll() is None
        receipt['finished_unix_s'] = time.time()
        save(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-cgroup', required=True, type=Path)
    parser.add_argument('--expected-parent-memory-max', required=True, type=int)
    parser.add_argument('--declared-host-kv-offload-bytes', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interval-s', type=float, default=1.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    result = run_observed(command, parent=args.parent_cgroup,
        expected_parent_memory_max=args.expected_parent_memory_max, output=args.output,
        declared_host_kv_offload_bytes=args.declared_host_kv_offload_bytes, interval_s=args.interval_s)
    print(json.dumps({'status': result['status'], 'receipt': str(args.output/'terminal.json')}))
    return 0 if result['status'] == 'COMPLETE_OBSERVED_ONLY' else 1


if __name__ == '__main__':
    sys.exit(main())
