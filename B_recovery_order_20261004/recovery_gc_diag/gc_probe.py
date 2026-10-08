"""Host GC callback boundaries only; no collection or GC/runtime setting changes."""
import gc
import time


def _state():
    return dict(host_perf_s=time.perf_counter(), gc_enabled=gc.isenabled(),
                thresholds=list(gc.get_threshold()))


def install():
    data = dict(status='INSTALLED', installation=_state(), uninstallation=None,
        clock='time.perf_counter host callback entry; same clock as raw, not GPU time',
        scope='Retained events can be clipped offline to raw measurement origin/end; '
              'short-event aggregates cover the installation window only',
        timing_semantics='Start/stop are this appended callback observations, not exact '
                         'GC or OS suspension boundaries; no cause attribution',
        retention=dict(min_duration_s=0.001, always_keep_generation=2), events=[],
        per_generation=[dict(generation=g, starts=0, stops=0, paired=0,
            total_duration_s=0.0, short_count=0, short_duration_s=0.0,
            collected=0, uncollectable=0) for g in range(3)],
        pairing_issues=dict(invalid_phase_or_generation=0, repeated_start=0,
                            unmatched_stop=0), pending_at_uninstall=[])
    starts = [None, None, None]
    active = True

    def callback(phase, info):
        if not active:
            return
        now = time.perf_counter()
        generation = info.get('generation')
        if type(generation) is not int or not 0 <= generation < 3 or phase not in ('start', 'stop'):
            data['pairing_issues']['invalid_phase_or_generation'] += 1
            return
        totals = data['per_generation'][generation]
        if phase == 'start':
            totals['starts'] += 1
            if starts[generation] is not None:
                data['pairing_issues']['repeated_start'] += 1
            starts[generation] = now
            return
        totals['stops'] += 1
        begin = starts[generation]
        starts[generation] = None
        if begin is None:
            data['pairing_issues']['unmatched_stop'] += 1
            return
        duration = now-begin
        collected, uncollectable = info.get('collected', 0), info.get('uncollectable', 0)
        totals['paired'] += 1
        totals['total_duration_s'] += duration
        totals['collected'] += collected
        totals['uncollectable'] += uncollectable
        if duration >= 0.001 or generation == 2:
            data['events'].append(dict(generation=generation,
                start_host_perf_s=begin, stop_host_perf_s=now, duration_s=duration,
                collected=collected, uncollectable=uncollectable))
        else:
            totals['short_count'] += 1
            totals['short_duration_s'] += duration

    gc.callbacks.append(callback)

    def uninstall():
        nonlocal active
        if active:
            active = False
            try:
                gc.callbacks.remove(callback)
                data['callback_removed'] = True
            except ValueError:
                data['callback_removed'] = False
            data['uninstallation'] = _state()
            data['pending_at_uninstall'] = [dict(generation=g, start_host_perf_s=t)
                for g, t in enumerate(starts) if t is not None]
            data['status'] = 'UNINSTALLED'
        return data
    return data, uninstall
