"""One shared-lock acquisition, then a bounded first-boundary GPU handoff.

Only the initial boundary waits for an existing process or residual counters.
The frozen experiment, later boundaries, and whole-group budget are unchanged.
"""
import fcntl
import os
import signal
import subprocess
import time

import run_native_host_group as group
import run_native_host_group_wait as lock_wait


HANDOFF_SECONDS = 300
POLL_SECONDS = 5


class HandoffExpired(TimeoutError):
    pass


def initial_boundary(uuid, read_state, *, clock=time.monotonic, sleep=time.sleep):
    """Read the original GPU state while the base controller retains its flock."""
    started = clock()
    deadline = started + HANDOFF_SECONDS
    readings, attempts, timed_out, timeout_reason = [], 0, False, None
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    group_expires_first = 0 < previous_remaining <= HANDOFF_SECONDS

    def expired(number, frame):
        if group_expires_first:
            if callable(previous_handler):
                previous_handler(number, frame)
            raise TimeoutError('Original group wall budget exhausted during handoff')
        raise HandoffExpired('Initial GPU handoff exceeded 300 seconds')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL,
                     previous_remaining if group_expires_first else HANDOFF_SECONDS)
    try:
        while clock() < deadline:
            attempts += 1
            state = read_state(uuid)
            readings.append(state)
            if clock() >= deadline:
                timed_out = True
                break
            if state['empty'] is True:
                break
            reason = 'compute_processes' if state['process_rows'] else 'residual_counters'
            print(f'WAIT_GPU_HANDOFF controller_pid={os.getpid()} reason={reason} '
                  f'elapsed_seconds={clock()-started:.3f} '
                  f'memory_used_mib={state["memory_used_mib"]} '
                  f'utilization_gpu_percent={state["utilization_gpu_percent"]}', flush=True)
            # Use gpu_state rather than gpu_boundary: its built-in two-second
            # reread would violate this handoff's five-second polling interval.
            sleep(min(POLL_SECONDS, max(0.0, deadline - clock())))
        else:
            timed_out = True
    except (HandoffExpired, subprocess.TimeoutExpired) as error:
        timed_out = True
        timeout_reason = f'{type(error).__name__}: {error}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        # The wait consumes the original group budget; never restart its clock.
        remaining = previous_remaining - (clock() - started)
        if previous_remaining > 0 and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)

    elapsed = clock() - started
    result = dict(readings[-1]) if readings else dict(
        authorized_gpu_uuid=uuid, uuid=None, name=None, empty=False,
        identity_status='UNKNOWN_NO_COMPLETE_GPU_STATE')
    if timed_out:
        # A read finishing at/after the deadline cannot authorize CUDA startup.
        result['empty'] = False
    result.update(boundary_readings=readings, handoff_wait_wall_s=elapsed,
                  handoff_limit_seconds=HANDOFF_SECONDS, handoff_poll_seconds=POLL_SECONDS,
                  handoff_poll_attempts=attempts, handoff_timed_out=timed_out,
                  handoff_timeout_reason=timeout_reason)
    status = 'GPU_HANDOFF_TIMEOUT' if timed_out else 'GPU_HANDOFF_READY'
    print(f'{status} controller_pid={os.getpid()} elapsed_seconds={elapsed:.3f} '
          f'complete_readings={len(readings)} no_retry=true', flush=True)
    # On timeout the frozen base writes this initial boundary then its original
    # require(initial['empty']) fails. No CUDA initialization can follow it.
    return result


def first_boundary_only(original_boundary, read_state, *, clock=time.monotonic, sleep=time.sleep):
    first = True

    def boundary(uuid):
        nonlocal first
        if not first:
            return original_boundary(uuid)
        first = False
        return initial_boundary(uuid, read_state, clock=clock, sleep=sleep)

    return boundary


def main():
    print(f'NATIVE_HOST_HANDOFF_ENTRY controller_pid={os.getpid()} '
          f'entry_sha256={group.base.digest(__file__)} '
          f'handoff_limit_seconds={HANDOFF_SECONDS}', flush=True)
    original_flock = fcntl.flock
    original_boundary = group.base.gpu_boundary
    fcntl.flock = lock_wait.acquire_once
    group.base.gpu_boundary = first_boundary_only(original_boundary, group.base.gpu_state)
    try:
        return group.main()
    finally:
        group.base.gpu_boundary = original_boundary
        fcntl.flock = original_flock


if __name__ == '__main__':
    raise SystemExit(main())
