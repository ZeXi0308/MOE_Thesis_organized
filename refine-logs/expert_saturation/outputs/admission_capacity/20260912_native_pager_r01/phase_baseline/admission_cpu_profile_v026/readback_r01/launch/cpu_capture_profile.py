"""Current-thread CPU profiling around measurement capture only."""
import cProfile
import json
import time


def profile_capture(capture, output, *, enabled, tag):
    profile = cProfile.Profile(timer=time.thread_time)
    meta = dict(tag=tag, profile_enabled=bool(enabled), timer="time.thread_time",
        wrapper_start_wall_s=time.perf_counter(), wrapper_start_cpu_s=time.thread_time(),
        capture_returned=False, statistics_saved=False,
        scope="Instrumented capture diagnostic; profile tax remains in raw wall; dump_stats occurs after capture")
    result = None
    try:
        if enabled:
            profile.enable()
        meta.update(capture_start_wall_s=time.perf_counter(), capture_start_cpu_s=time.thread_time())
        result = capture()
        meta.update(capture_returned=True, capture_status=result.get("status"),
                    raw_capture_wall_s=result.get("observation_end_s"))
        return result
    except BaseException as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta.update(capture_end_wall_s=time.perf_counter(), capture_end_cpu_s=time.thread_time())
        if enabled:
            profile.disable()
        meta.update(profile_disabled=True, wrapper_end_wall_s=time.perf_counter(),
                    wrapper_end_cpu_s=time.thread_time())
        if enabled:
            stats_path = output / "capture_thread_cpu.pstats"
            profile.dump_stats(str(stats_path))
            meta.update(statistics_saved=True, statistics_file=stats_path.name)
        (output / "cpu_profile.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
