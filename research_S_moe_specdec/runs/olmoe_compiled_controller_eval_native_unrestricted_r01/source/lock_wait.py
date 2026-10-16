"""Bounded kernel flock wait for the Unix campaign main thread.

The caller supplies a newly opened, unlocked file and retains ownership on True.
False means the deadline expired and this descriptor holds no lock. No helper
process or background waiter survives the call. An existing SIGALRM timer or
blocked SIGALRM is rejected rather than silently replacing another deadline.
"""
import fcntl
import math
import signal
import threading
import time


class _DeadlineExpired(Exception):
    pass


def acquire_exclusive(lock, timeout_seconds):
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('lock timeout must be finite and nonnegative')
    if timeout == 0:
        return False
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('bounded flock wait must run in the main thread')
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    previous_handler = signal.getsignal(signal.SIGALRM)
    installed = acquired = False

    def expire(signum, frame):
        # Raising interrupts flock instead of allowing Python's EINTR retry.
        raise _DeadlineExpired()

    try:
        if (signal.SIGALRM in previous_mask or any(signal.getitimer(signal.ITIMER_REAL))
                or signal.SIGALRM in signal.sigpending()):
            raise RuntimeError('bounded flock wait requires an unused, unblocked SIGALRM')
        signal.signal(signal.SIGALRM, expire)
        installed = True
        deadline = time.monotonic() + timeout
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            fcntl.flock(lock, fcntl.LOCK_EX)
            acquired = True
        except _DeadlineExpired:
            acquired = False
        finally:
            # A deadline can race with successful acquisition or with entering
            # cleanup. Block and consume our pending alarm before restoring the
            # caller's handler; never retain a lock obtained after the deadline.
            try:
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            except _DeadlineExpired:
                acquired = False
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            signal.setitimer(signal.ITIMER_REAL, 0)
            if signal.SIGALRM in signal.sigpending():
                signal.sigwait({signal.SIGALRM})
                acquired = False
            if time.monotonic() >= deadline:
                acquired = False
        return acquired
    finally:
        try:
            if installed:
                if not acquired:
                    # Also releases the acquisition/timeout boundary race.
                    fcntl.flock(lock, fcntl.LOCK_UN)
                signal.signal(signal.SIGALRM, previous_handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
