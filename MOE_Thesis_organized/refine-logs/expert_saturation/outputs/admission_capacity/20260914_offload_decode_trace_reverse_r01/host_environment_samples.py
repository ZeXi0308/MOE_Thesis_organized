"""Read-only coarse host observations in the driver, identical for both arms."""
import json, os, subprocess, time
from pathlib import Path


def snapshot(pid):
    row = dict(wall_s=time.time(), monotonic_s=time.monotonic())
    for key, path in [('loadavg','/proc/loadavg'), ('cpu_stat','/proc/stat'),
                      ('process_stat',f'/proc/{pid}/stat'),
                      ('process_status',f'/proc/{pid}/status'),
                      ('cgroup_cpu_stat','/sys/fs/cgroup/cpu.stat'),
                      ('cgroup_cpu_max','/sys/fs/cgroup/cpu.max')]:
        try: row[key] = Path(path).read_text()
        except OSError as e: row[key] = dict(error=str(e))
    try:
        row['affinity'] = sorted(os.sched_getaffinity(pid))
        # /proc values are observations, not controlled CPU frequency.
        row['cpu_mhz'] = [s for s in Path('/proc/cpuinfo').read_text().splitlines()
                          if s.startswith('cpu MHz')]
    except (OSError, AttributeError) as e: row['cpu_observation_error'] = str(e)
    return row


def run_sampled(command, *, cwd, env, log, output, timeout):
    rows = []
    started = time.monotonic()
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                          stderr=subprocess.STDOUT) as proc:
        try:
            while True:
                rows.append(snapshot(proc.pid))
                try:
                    proc.wait(timeout=min(1., max(.001, timeout-(time.monotonic()-started))))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started >= timeout:
                        proc.kill()  # Only this driver's own child.
                        proc.wait()
                        raise
            return subprocess.CompletedProcess(command, proc.returncode)
        finally:
            Path(output).write_text(json.dumps(dict(pid=proc.pid, samples=rows,
                clock_ticks=os.sysconf('SC_CLK_TCK'), interval_s=1,
                scope='Driver read-only sampling; coarse host context, not per-call attribution or frequency control.'), indent=2)+'\n')
