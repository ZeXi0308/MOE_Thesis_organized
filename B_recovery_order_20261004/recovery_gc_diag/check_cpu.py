"""Small deterministic callback-pairing and real GC lifecycle checks; no collect."""
from types import SimpleNamespace as NS

import gc_probe

real_gc, real_clock = gc_probe.gc, gc_probe.time
sentinel = lambda phase, info: None
later = lambda phase, info: None
fake_gc = NS(callbacks=[sentinel], isenabled=lambda: True,
             get_threshold=lambda: (700, 10, 10))
tick = [10.0]
gc_probe.gc = fake_gc
gc_probe.time = NS(perf_counter=lambda: tick[0])
try:
    data, uninstall = gc_probe.install()
    callback = fake_gc.callbacks[-1]
    assert fake_gc.callbacks == [sentinel, callback]
    def emit(phase, generation, at, **info):
        tick[0] = at
        callback(phase, dict(generation=generation, **info))
    emit('start', 0, 10.1); emit('stop', 0, 10.1005, collected=2)
    emit('start', 1, 10.2); emit('stop', 1, 10.2015, collected=3, uncollectable=1)
    emit('start', 2, 10.3); emit('stop', 2, 10.3002)
    assert [e['generation'] for e in data['events']] == [1, 2]
    assert data['events'][0]['start_host_perf_s'] == 10.2
    assert data['events'][0]['stop_host_perf_s'] == 10.2015
    assert data['per_generation'][0]['short_count'] == 1
    assert abs(data['per_generation'][0]['short_duration_s']-.0005) < 1e-10
    assert [g['paired'] for g in data['per_generation']] == [1, 1, 1]
    assert [g['collected'] for g in data['per_generation']] == [2, 3, 0]
    emit('stop', 0, 10.4)
    emit('start', 1, 10.5); emit('start', 1, 10.6)
    emit('unknown', 7, 10.7)
    assert data['pairing_issues'] == dict(invalid_phase_or_generation=1,
                                        repeated_start=1, unmatched_stop=1)
    fake_gc.callbacks.append(later)
    tick[0] = 10.8
    assert uninstall() is data and uninstall() is data
    assert fake_gc.callbacks == [sentinel, later]
    assert data['pending_at_uninstall'] == [dict(generation=1, start_host_perf_s=10.6)]
    assert data['uninstallation'] == dict(host_perf_s=10.8, gc_enabled=True,
                                        thresholds=[700, 10, 10])
    callback('stop', {'generation': 1})
    assert len(data['events']) == 2  # A stale callback cannot log after uninstall.
finally:
    gc_probe.gc, gc_probe.time = real_gc, real_clock

# Real module: retain all pre-existing callbacks and all settings, without
# triggering collection. finally-uninstall does not consume the caller's error.
before_callbacks = list(real_gc.callbacks)
before_settings = (real_gc.isenabled(), real_gc.get_threshold())
data, uninstall = gc_probe.install()
assert real_gc.callbacks[:-1] == before_callbacks
failure = RuntimeError('original runtime failure')
try:
    try:
        raise failure
    finally:
        uninstall()
except RuntimeError as error:
    assert error is failure
assert uninstall() is data and real_gc.callbacks == before_callbacks
assert before_settings == (real_gc.isenabled(), real_gc.get_threshold())
assert data['installation']['gc_enabled'] == data['uninstallation']['gc_enabled']
assert data['installation']['thresholds'] == data['uninstallation']['thresholds']
print('PASS: paired host timestamps; >=1ms/gen2 retention; short counters; unmatched/pending visibility; preserve callbacks/settings; idempotent uninstall; original error propagates')
