"""Passive 1 Hz GPU and bounded parent-driver CPU snapshots.

Never initializes CUDA or changes affinity, scheduling, clocks, or cgroups.
Host fields are diagnostic only; missing Linux files never stop the observer.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time


def parse_proc_stat(text):
    """Linux stat field numbers, with comm allowed to contain spaces or ')'."""
    left, right = text.find('('), text.rfind(')')
    if left < 1 or right < left:
        raise ValueError('invalid proc stat comm delimiters')
    fields = text[right + 1:].split()  # First item is field 3 (state).
    if len(fields) < 37:
        raise ValueError('proc stat has no processor field 39')
    result = dict(pid=int(text[:left].strip()), comm=text[left + 1:right], state=fields[0])
    for name, number in [('ppid', 4), ('utime_ticks', 14), ('stime_ticks', 15),
                         ('num_threads', 20), ('starttime_ticks', 22), ('processor', 39)]:
        result[name] = int(fields[number - 3])
    return result


def parse_proc_status(text):
    fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
    if not all(key in fields for key in ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches')):
        raise ValueError('proc status has no context-switch counters')
    result = {key: int(fields[key].strip()) for key in
              ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches')}
    result['cpus_allowed_list'] = fields.get('Cpus_allowed_list', '').strip() or None
    return result


def parse_cpu_stat(text):
    """cgroup cpu.stat counters; units stay in their native field names."""
    result = {key: int(value) for key, value in (line.split() for line in text.splitlines() if line.strip())}
    if not result:
        raise ValueError('empty cgroup cpu.stat')
    return result


def parse_cpu_pressure(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, *pairs = line.split()
        values = dict(pair.split('=', 1) for pair in pairs)
        if not all(key in values for key in ('avg10', 'avg60', 'avg300', 'total')):
            raise ValueError('incomplete CPU pressure counters')
        result[name] = {key: int(value) if key == 'total' else float(value)
                        for key, value in values.items()}
    if not result:
        raise ValueError('empty CPU pressure data')
    return result


def _mount_unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)


def resolve_cgroup(membership_text, mountinfo_text):
    """Map a namespace-visible membership through a matching visible mount.

    Never assume /sys/fs/cgroup is the hierarchy root or append an unverified
    host path. Parent and observer namespaces must also match at the caller.
    """
    members = []
    for line in membership_text.splitlines():
        if not line.strip():
            continue
        hierarchy, controllers, member = line.split(':', 2)
        if hierarchy == '0' and not controllers:
            members.append(('v2', member))
        elif 'cpu' in controllers.split(','):
            members.append(('v1', member))
    mounts = []
    for line in mountinfo_text.splitlines():
        fields = line.split()
        if '-' not in fields:
            continue
        sep = fields.index('-')
        if len(fields) <= sep + 3 or len(fields) < 6:
            continue
        fs_type = fields[sep + 1]
        if fs_type == 'cgroup2':
            version = 'v2'
        elif fs_type == 'cgroup' and 'cpu' in fields[sep + 3].split(','):
            version = 'v1'
        else:
            continue
        mounts.append((version, _mount_unescape(fields[3]), _mount_unescape(fields[4])))
    candidates = []
    for version, member in members:
        if not member.startswith('/') or '..' in PurePosixPath(member).parts:
            raise ValueError('cgroup membership escapes visible root')
        for mount_version, root, mount in mounts:
            if version != mount_version or not root.startswith('/') or not mount.startswith('/'):
                continue
            if '..' in PurePosixPath(root).parts or '..' in PurePosixPath(mount).parts:
                continue
            if member == root:
                relative = ''
            elif root == '/':
                relative = member.lstrip('/')
            elif member.startswith(root.rstrip('/') + '/'):
                relative = member[len(root):].lstrip('/')
            else:
                continue
            candidates.append(dict(version=version, member_path=member, mount_root=root,
                mount_point=mount, directory=str(Path(mount) / relative)))
    if not candidates:
        raise ValueError('no matching namespace-visible CPU cgroup mount')
    # On a hybrid hierarchy an explicit v1 CPU controller owns CPU throttling;
    # a generic v2 membership must not silently replace it.
    return max(candidates, key=lambda x: (x['version'] == 'v1', len(x['mount_root'])))


def _text(path, limit=8192):
    try:
        with Path(path).open() as stream:
            value = stream.read(limit + 1)
        if len(value) > limit:
            raise ValueError(f'file exceeds {limit}-character read limit')
        return value, None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _parsed(path, parser, limit=8192):
    text, error = _text(path, limit)
    values = None
    if error is None:
        try:
            values = parser(text)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return dict(path=str(path), values=values, error=error)


def _link(path):
    try:
        return dict(path=str(path), value=os.readlink(path), error=None)
    except Exception as exc:
        return dict(path=str(path), value=None, error=f'{type(exc).__name__}: {exc}')


class HostCpuProbe:
    def __init__(self, parent_pid, proc_root=Path('/proc'), sys_root=Path('/sys')):
        self.parent_pid = parent_pid
        self.proc_root, self.sys_root = Path(proc_root), Path(sys_root)
        self.parent = self.proc_root / str(parent_pid)
        self.task = self.parent / 'task' / str(parent_pid)  # Main driver TID == PID.
        self.membership_path = self.parent / 'cgroup'
        try:
            self.clock_ticks = os.sysconf('SC_CLK_TCK')
            self.clock_ticks_error = None
        except Exception as exc:
            self.clock_ticks, self.clock_ticks_error = None, repr(exc)
        membership, member_error = _text(self.membership_path, 4096)
        mountinfo_path = self.proc_root / 'self' / 'mountinfo'
        mountinfo, mount_error = _text(mountinfo_path, 65536)
        self.namespaces = {name: _link(self.proc_root / 'self' / 'ns' / name)
                           for name in ('cgroup', 'mnt')}
        self.membership, self.cgroup_mapping = membership, None
        error = member_error or mount_error
        if error is None:
            try:
                self.cgroup_mapping = resolve_cgroup(membership, mountinfo)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        self.cgroup_provenance = dict(membership_path=str(self.membership_path),
            initial_membership=membership, mountinfo_path=str(mountinfo_path),
            observer_namespaces=self.namespaces, mapping=self.cgroup_mapping, error=error,
            mapping_cached_at_start=True)

    def _frequency(self, processor):
        result = dict(cpu=processor, khz=None, source_path=None, error=None, attempts=[])
        if processor is None or processor < 0:
            result['error'] = 'driver last CPU unavailable'
            return result
        base = self.sys_root / 'devices' / 'system' / 'cpu' / f'cpu{processor}' / 'cpufreq'
        for name in ('scaling_cur_freq', 'cpuinfo_cur_freq'):
            path = base / name
            value = _parsed(path, lambda text: int(text.strip()), 128)
            if value['error'] is None and value['values'] > 0:
                result.update(khz=value['values'], source_path=str(path))
                return result
            result['attempts'].append(dict(path=str(path),
                error=value['error'] or 'frequency is not positive'))
        result['error'] = 'current CPU frequency unavailable; no value inferred'
        return result

    def sample(self):
        start = time.perf_counter()
        stat = _parsed(self.task / 'stat', parse_proc_stat)
        status = _parsed(self.task / 'status', parse_proc_status, 16384)
        processor = stat['values']['processor'] if stat['values'] else None
        frequency = self._frequency(processor)
        membership, member_error = _text(self.membership_path, 4096)
        namespaces = {name: _link(self.parent / 'ns' / name) for name in ('cgroup', 'mnt')}
        error = self.cgroup_provenance['error'] or member_error
        if error is None and membership != self.membership:
            error = 'parent cgroup membership changed; cached mapping not reused'
        if error is None and any(not self.namespaces[name]['value'] or
                namespaces[name]['value'] != self.namespaces[name]['value']
                for name in ('cgroup', 'mnt')):
            error = 'parent/observer cgroup or mount namespace differs or is unreadable'
        unavailable = lambda reason: dict(path=None, values=None, error=reason)
        if error is None:
            group = Path(self.cgroup_mapping['directory'])
            cpu_stat = _parsed(group / 'cpu.stat', parse_cpu_stat)
            pressure = (_parsed(group / 'cpu.pressure', parse_cpu_pressure)
                        if self.cgroup_mapping['version'] == 'v2'
                        else unavailable('per-cgroup CPU pressure not supported here for v1'))
        else:
            cpu_stat, pressure = unavailable(error), unavailable(error)
        return dict(schema='E.passive_host_cpu.v1', monotonic_s=start,
            parent_pid=self.parent_pid, driver_tid=self.parent_pid,
            clock_ticks_per_second=self.clock_ticks, clock_ticks_error=self.clock_ticks_error,
            driver_task_stat=stat, driver_task_status=status, current_cpu_frequency=frequency,
            cgroup=dict(provenance=self.cgroup_provenance, current_membership=membership,
                parent_namespaces=namespaces, cpu_stat=cpu_stat, cpu_pressure=pressure),
            probe_wall_s=time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-pid', type=int, required=True)
    a = p.parse_args()
    try:
        host_probe, host_init_error = HostCpuProbe(a.parent_pid), None
    except Exception as exc:
        host_probe, host_init_error = None, repr(exc)
    while os.getppid() == a.parent_pid:
        start = time.perf_counter()
        row = dict(monotonic_s=start, unix_s=time.time(), observer_pid=os.getpid())
        for name, query in (
            ('gpu', '--query-gpu=uuid,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used'),
            ('compute_processes', '--query-compute-apps=pid,process_name,used_gpu_memory'),
        ):
            try:
                result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=3)
                row[name] = dict(returncode=result.returncode, stdout=result.stdout,
                                 stderr=result.stderr)
            except Exception as exc:
                row[name] = dict(error=repr(exc))
        row['query_wall_s'] = time.perf_counter() - start  # Existing GPU-query span.
        try:
            row['host_cpu'] = host_probe.sample() if host_probe else dict(error=host_init_error)
        except Exception as exc:
            row['host_cpu'] = dict(schema='E.passive_host_cpu.v1', error=repr(exc))
        row['sample_wall_s'] = time.perf_counter() - start
        print(json.dumps(row), flush=True)
        time.sleep(max(0, 1 - row['sample_wall_s']))


if __name__ == '__main__':
    main()
